//! Keeping a tool loop's history inside the context window.
//!
//! - [`cap_output`] shortens one tool output to its start and end, and says
//!   how much was left out.
//! - [`HistoryCompactor`] shrinks a whole history to a token budget without
//!   losing the task: the user's request is pinned, recent tool exchanges stay
//!   verbatim, older ones are condensed into evidence notes, and only then are
//!   the oldest turns digested away. With [`HistoryCompactor::compact_with_model`]
//!   a model writes the condensed notes instead.
//!
//! Token counts are estimates (four characters per token, a fixed cost per
//! image). The provider stays the authority on what fits; pair the compactor
//! with [`dcpr_llm::Error::is_context_overflow`] for the cases it misjudges.

use dcpr_llm::{ChatModel, Completion, ToolCall, ToolTurn};
use std::borrow::Cow;

/// Estimated tokens for one image.
const IMAGE_TOKENS: usize = 1_000;

fn text_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

/// Estimated tokens for one turn.
pub fn turn_tokens(turn: &ToolTurn) -> usize {
    match turn {
        ToolTurn::User(text) | ToolTurn::Assistant(text) => text_tokens(text),
        ToolTurn::UserWithImages(text, images) => text_tokens(text) + images.len() * IMAGE_TOKENS,
        ToolTurn::AssistantToolCalls(calls) => calls
            .iter()
            .map(|call| text_tokens(&call.name) + text_tokens(&call.arguments.to_string()))
            .sum(),
        ToolTurn::ToolResults(results) => results.iter().map(|(_, out)| text_tokens(out)).sum(),
    }
}

/// Estimated tokens for a history.
pub fn estimate_tokens(history: &[ToolTurn]) -> usize {
    history.iter().map(turn_tokens).sum()
}

/// `text` cut to at most about `max_chars` characters, keeping its start and
/// end and marking what was left out. Cuts fall on line breaks when one is
/// near, so log lines and table rows stay whole. Short text is returned as is.
pub fn cap_output(text: &str, max_chars: usize) -> Cow<'_, str> {
    let total = text.chars().count();
    if total <= max_chars {
        return Cow::Borrowed(text);
    }
    let head_chars = max_chars * 2 / 3;
    let tail_chars = max_chars - head_chars;
    let byte_at = |chars: usize| {
        text.char_indices()
            .nth(chars)
            .map_or(text.len(), |(i, _)| i)
    };
    let mut head_end = byte_at(head_chars);
    let mut tail_start = byte_at(total - tail_chars);
    // Snap to a line break within the last (first) fifth of each kept part.
    let slack_head = byte_at(head_chars - head_chars / 5);
    if let Some(newline) = text[slack_head..head_end].rfind('\n') {
        head_end = slack_head + newline + 1;
    }
    let slack_tail = byte_at(total - tail_chars + tail_chars / 5);
    if let Some(newline) = text[tail_start..slack_tail].find('\n') {
        tail_start += newline + 1;
    }
    let dropped = text[head_end..tail_start].chars().count();
    Cow::Owned(format!(
        "{}\n[… {dropped} of {total} characters omitted …]\n{}",
        &text[..head_end],
        &text[tail_start..]
    ))
}

/// What a compaction did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompactionReport {
    pub tokens_before: usize,
    pub tokens_after: usize,
    /// Tool outputs shortened with [`cap_output`].
    pub capped_outputs: usize,
    /// Tool exchanges (a call batch and its results) condensed into notes.
    pub folded_exchanges: usize,
    /// Older turns digested into a summary note.
    pub digested_turns: usize,
    /// Where the pinned task turn is now, when one was given.
    pub task_index: Option<usize>,
    /// Notes the compaction wrote, with the material each one condenses.
    notes: Vec<(usize, String)>,
}

impl CompactionReport {
    /// Whether the history changed.
    pub fn changed(&self) -> bool {
        self.capped_outputs + self.folded_exchanges + self.digested_turns > 0
    }
}

/// Shrinks a history to a token budget. See the module documentation.
#[derive(Debug, Clone)]
pub struct HistoryCompactor {
    /// Compact when the estimate is above this, and aim below it.
    pub max_tokens: usize,
    /// Most recent tool exchanges kept verbatim.
    pub keep_recent_exchanges: usize,
    /// Older tool outputs are capped to this many characters first.
    pub stale_output_chars: usize,
    /// Each call in a condensed note keeps this much of its result.
    pub evidence_chars: usize,
}

impl HistoryCompactor {
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            keep_recent_exchanges: 3,
            stale_output_chars: 2_000,
            evidence_chars: 400,
        }
    }

    pub fn keep_recent_exchanges(mut self, exchanges: usize) -> Self {
        self.keep_recent_exchanges = exchanges;
        self
    }

    /// Shrink `history` in place until its estimate is within the budget, or
    /// nothing more can go. `task` is the index of the turn that states the
    /// current request; it is never altered or removed. In order, stopping as
    /// soon as the history fits:
    ///
    /// 1. Cap tool outputs older than the recent exchanges.
    /// 2. Condense those older exchanges into one evidence note per run, which
    ///    keeps what each tool was asked and the start and end of its answer.
    /// 3. Digest the oldest remaining turns, before and after the task, into
    ///    one note each side.
    /// 4. Cap the recent outputs too.
    ///
    /// A results turn is never separated from its calls.
    pub fn compact(&self, history: &mut Vec<ToolTurn>, task: Option<usize>) -> CompactionReport {
        let mut report = CompactionReport {
            tokens_before: estimate_tokens(history),
            task_index: task,
            ..Default::default()
        };
        let fits = |history: &[ToolTurn]| estimate_tokens(history) <= self.max_tokens;
        if !fits(history) {
            self.cap_stale_outputs(history, &mut report);
        }
        if !fits(history) {
            self.fold_stale_exchanges(history, &mut report);
        }
        if !fits(history) {
            self.digest_oldest(history, &mut report);
        }
        if !fits(history) {
            for turn in history.iter_mut() {
                report.capped_outputs += cap_results(turn, self.stale_output_chars);
            }
        }
        report.tokens_after = estimate_tokens(history);
        report
    }

    /// Like [`compact`](Self::compact), but `model` rewrites each note in its
    /// own words, keeping the facts the agent still needs. A note whose
    /// summary fails keeps its mechanical form.
    pub async fn compact_with_model<C, M>(
        &self,
        model: &M,
        caller: &C,
        history: &mut Vec<ToolTurn>,
        task: Option<usize>,
    ) -> CompactionReport
    where
        C: ?Sized + Sync,
        M: ChatModel<C> + ?Sized,
    {
        let mut report = self.compact(history, task);
        let task_text = report
            .task_index
            .and_then(|index| history.get(index))
            .and_then(|turn| match turn {
                ToolTurn::User(text) | ToolTurn::UserWithImages(text, _) => Some(text.as_str()),
                _ => None,
            })
            .unwrap_or("(not given)")
            .to_string();
        for (index, source) in std::mem::take(&mut report.notes) {
            let system = format!(
                "You condense an AI agent's earlier conversation so it can continue its task. \
                 Keep every fact, number, identifier, file path, error and outcome the agent may \
                 still need, and say which tool produced each. Drop pleasantries and repetition. \
                 Plain text, at most {} words.\n\nThe agent's task: {task_text}",
                self.evidence_chars.max(200) / 2
            );
            let request = Completion::new(&system, &source, 1_024);
            match model.complete(caller, request).await {
                Ok(summary) if !summary.trim().is_empty() => {
                    if let Some(ToolTurn::Assistant(text) | ToolTurn::User(text)) =
                        history.get_mut(index)
                    {
                        let header = text.lines().next().unwrap_or_default().to_string();
                        *text = format!("{header}\n{}", summary.trim());
                    }
                }
                Ok(_) => {}
                Err(error) => tracing::warn!("history summary failed, keeping the digest: {error}"),
            }
        }
        report.tokens_after = estimate_tokens(history);
        report
    }

    /// Indices of the call turns of the most recent exchanges.
    fn recent_exchange_starts(&self, history: &[ToolTurn]) -> Vec<usize> {
        let starts = exchange_starts(history);
        let keep = starts.len().saturating_sub(self.keep_recent_exchanges);
        starts[keep..].to_vec()
    }

    fn cap_stale_outputs(&self, history: &mut [ToolTurn], report: &mut CompactionReport) {
        let recent = self.recent_exchange_starts(history);
        let first_recent = recent.first().copied().unwrap_or(history.len());
        for turn in &mut history[..first_recent] {
            report.capped_outputs += cap_results(turn, self.stale_output_chars);
        }
    }

    fn fold_stale_exchanges(&self, history: &mut Vec<ToolTurn>, report: &mut CompactionReport) {
        let recent = self.recent_exchange_starts(history);
        let first_recent = recent.first().copied().unwrap_or(history.len());
        let mut out = Vec::with_capacity(history.len());
        let mut index = 0;
        let mut new_task = report.task_index;
        let old = std::mem::take(history);
        while index < old.len() {
            let stale_exchange =
                index < first_recent && matches!(old[index], ToolTurn::AssistantToolCalls(_));
            if !stale_exchange {
                if Some(index) == report.task_index {
                    new_task = Some(out.len());
                }
                out.push(old[index].clone());
                index += 1;
                continue;
            }
            // A run of consecutive stale exchanges becomes one note.
            let mut lines = Vec::new();
            let mut source = Vec::new();
            while index < first_recent {
                let ToolTurn::AssistantToolCalls(calls) = &old[index] else {
                    break;
                };
                let results = match old.get(index + 1) {
                    Some(ToolTurn::ToolResults(results)) => results.as_slice(),
                    _ => &[],
                };
                for call in calls {
                    let output = results
                        .iter()
                        .find(|(id, _)| *id == call.id)
                        .map_or("(no result)", |(_, output)| output.as_str());
                    lines.push(evidence_line(call, output, self.evidence_chars));
                    source.push(evidence_line(call, output, self.stale_output_chars));
                }
                report.folded_exchanges += 1;
                index += if results.is_empty() { 1 } else { 2 };
            }
            const HEADER: &str = "[Earlier tool results, condensed]";
            // Extend the note an earlier compaction left just before this run.
            if let Some(ToolTurn::Assistant(previous)) = out.last_mut()
                && previous.starts_with(HEADER)
            {
                previous.push('\n');
                previous.push_str(&lines.join("\n"));
                let at = out.len() - 1;
                match report.notes.iter_mut().find(|(index, _)| *index == at) {
                    Some((_, existing)) => {
                        existing.push('\n');
                        existing.push_str(&source.join("\n"));
                    }
                    None => report.notes.push((at, source.join("\n"))),
                }
            } else {
                report.notes.push((out.len(), source.join("\n")));
                out.push(ToolTurn::Assistant(format!(
                    "{HEADER}\n{}",
                    lines.join("\n")
                )));
            }
        }
        report.task_index = new_task;
        *history = out;
    }

    /// Digest the oldest turns on each side of the task into one note,
    /// keeping the task and the recent exchanges.
    fn digest_oldest(&self, history: &mut Vec<ToolTurn>, report: &mut CompactionReport) {
        let keep_from = self
            .recent_exchange_starts(history)
            .first()
            .copied()
            .unwrap_or(history.len().saturating_sub(1));
        // Ranges to digest: before the task, and between it and the recent
        // tail. Without a task, everything before the tail.
        let ranges: Vec<(usize, usize)> = match report.task_index {
            Some(task) if task < keep_from => vec![(0, task), (task + 1, keep_from)],
            Some(task) => vec![(0, task.min(keep_from))],
            None => vec![(0, keep_from)],
        };
        let old = std::mem::take(history);
        // Where each surviving old turn lands.
        let mut moved: Vec<Option<usize>> = vec![None; old.len()];
        let mut out = Vec::with_capacity(old.len());
        let mut digests = Vec::new();
        let mut index = 0;
        for (start, end) in ranges {
            while index < start.min(old.len()) {
                moved[index] = Some(out.len());
                out.push(old[index].clone());
                index += 1;
            }
            // Never leave a results turn without its calls.
            let mut end = end.min(old.len());
            while end < old.len() && matches!(old[end], ToolTurn::ToolResults(_)) {
                end += 1;
            }
            if end <= index {
                continue;
            }
            let lines = |max_chars| {
                old[index..end]
                    .iter()
                    .map(|turn| digest_line(turn, max_chars))
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            // Before the task the note stands in for the user's earlier
            // conversation; after it, for the agent's own work.
            let header = if report.task_index.is_some_and(|task| index > task) {
                "[Earlier work on this request, condensed]"
            } else {
                "[Earlier conversation, condensed]"
            };
            let note = format!("{header}\n{}", lines(self.evidence_chars));
            report.digested_turns += end - index;
            digests.push((out.len(), lines(self.stale_output_chars)));
            out.push(if out.is_empty() {
                ToolTurn::User(note)
            } else {
                ToolTurn::Assistant(note)
            });
            index = end;
        }
        while index < old.len() {
            moved[index] = Some(out.len());
            out.push(old[index].clone());
            index += 1;
        }
        report.task_index = report.task_index.and_then(|task| moved[task]);
        // Earlier notes that were digested are covered by the new ones.
        report.notes = std::mem::take(&mut report.notes)
            .into_iter()
            .filter_map(|(at, source)| moved.get(at).copied().flatten().map(|to| (to, source)))
            .chain(digests)
            .collect();
        *history = out;
    }
}

fn exchange_starts(history: &[ToolTurn]) -> Vec<usize> {
    history
        .iter()
        .enumerate()
        .filter(|(_, turn)| matches!(turn, ToolTurn::AssistantToolCalls(_)))
        .map(|(index, _)| index)
        .collect()
}

fn cap_results(turn: &mut ToolTurn, max_chars: usize) -> usize {
    let ToolTurn::ToolResults(results) = turn else {
        return 0;
    };
    let mut capped = 0;
    for (_, output) in results {
        if let Some(short) = cap_result(output, max_chars) {
            *output = short;
            capped += 1;
        }
    }
    capped
}

/// A shorter tool result, or `None` when it already fits. A [`ToolResult`]
/// envelope keeps its JSON shape: its message is capped instead of the text.
///
/// [`ToolResult`]: crate::ToolResult
fn cap_result(output: &str, max_chars: usize) -> Option<String> {
    if output.chars().count() <= max_chars {
        return None;
    }
    if let Ok(serde_json::Value::Object(mut envelope)) = serde_json::from_str(output)
        && let Some(message) = envelope.get("message").and_then(|m| m.as_str())
        && let Cow::Owned(short) = cap_output(message, max_chars)
    {
        envelope.insert("message".into(), short.into());
        return Some(serde_json::Value::Object(envelope).to_string());
    }
    match cap_output(output, max_chars) {
        Cow::Owned(short) => Some(short),
        Cow::Borrowed(_) => None,
    }
}

/// The part of a tool result worth keeping: for a [`ToolResult`] envelope its
/// status and message, otherwise the text itself.
///
/// [`ToolResult`]: crate::ToolResult
fn result_text(output: &str) -> Cow<'_, str> {
    let Ok(serde_json::Value::Object(envelope)) = serde_json::from_str(output) else {
        return Cow::Borrowed(output.trim());
    };
    let status = envelope.get("status").and_then(|s| s.as_str());
    let message = envelope.get("message").and_then(|m| m.as_str());
    match (status, message) {
        (Some(status), Some(message)) => Cow::Owned(format!("[{status}] {}", message.trim())),
        _ => Cow::Borrowed(output.trim()),
    }
}

fn evidence_line(call: &ToolCall, output: &str, max_chars: usize) -> String {
    let arguments = call.arguments.to_string();
    let arguments = cap_output(&arguments, 200);
    let text = result_text(output);
    let output = cap_output(&text, max_chars);
    format!("- {}({arguments}) → {output}", call.name)
}

const NOTE_PREFIX: &str = "[Earlier ";

/// The body of a note this module wrote, without its header line.
fn note_body(turn: &ToolTurn) -> Option<&str> {
    match turn {
        ToolTurn::User(text) | ToolTurn::Assistant(text) if text.starts_with(NOTE_PREFIX) => {
            Some(text.split_once('\n').map_or("", |(_, body)| body))
        }
        _ => None,
    }
}

fn digest_line(turn: &ToolTurn, max_chars: usize) -> String {
    // A note from an earlier compaction is already condensed.
    if let Some(body) = note_body(turn) {
        return body.to_string();
    }
    match turn {
        ToolTurn::User(text) => format!("- user: {}", cap_output(text.trim(), max_chars)),
        ToolTurn::UserWithImages(text, images) => format!(
            "- user ({} image(s)): {}",
            images.len(),
            cap_output(text.trim(), max_chars)
        ),
        ToolTurn::Assistant(text) => format!("- assistant: {}", cap_output(text.trim(), max_chars)),
        ToolTurn::AssistantToolCalls(calls) => calls
            .iter()
            .map(|call| {
                format!(
                    "- called {}({})",
                    call.name,
                    cap_output(&call.arguments.to_string(), 200)
                )
            })
            .collect::<Vec<_>>()
            .join("\n"),
        ToolTurn::ToolResults(results) => results
            .iter()
            .map(|(_, output)| format!("  → {}", cap_output(&result_text(output), max_chars)))
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dcpr_llm::ScriptedModel;
    use serde_json::json;

    fn exchange(n: usize, output: &str) -> [ToolTurn; 2] {
        let id = format!("c{n}");
        [
            ToolTurn::AssistantToolCalls(vec![ToolCall {
                id: id.clone(),
                name: "read_log".into(),
                arguments: json!({"n": n}),
            }]),
            ToolTurn::ToolResults(vec![(id, output.into())]),
        ]
    }

    fn results_follow_calls(history: &[ToolTurn]) -> bool {
        history.iter().enumerate().all(|(i, turn)| {
            !matches!(turn, ToolTurn::ToolResults(_))
                || matches!(
                    history.get(i.wrapping_sub(1)),
                    Some(ToolTurn::AssistantToolCalls(_))
                )
        })
    }

    #[test]
    fn capping_keeps_both_ends_and_says_what_was_dropped() {
        let text = (0..400).map(|i| format!("line {i}\n")).collect::<String>();
        let capped = cap_output(&text, 600);
        assert!(capped.starts_with("line 0\n"));
        assert!(capped.trim_end().ends_with("line 399"));
        assert!(capped.contains("characters omitted"), "{capped}");
        assert!(capped.chars().count() < 700);
        // Cuts fall on line breaks.
        let marker = capped.find("\n[…").unwrap();
        assert!(
            capped[..marker].ends_with('\n')
                || capped[..marker].ends_with(|c: char| c.is_ascii_digit())
        );
        assert!(matches!(cap_output("short", 600), Cow::Borrowed("short")));
        // Multibyte text is cut on character boundaries.
        let wide = "é".repeat(1_000);
        assert!(cap_output(&wide, 100).contains("characters omitted"));
    }

    #[test]
    fn under_budget_nothing_changes() {
        let mut history = vec![ToolTurn::User("task".into())];
        history.extend(exchange(0, "ok"));
        let before = history.clone();
        let report = HistoryCompactor::new(10_000).compact(&mut history, Some(0));
        assert!(!report.changed());
        assert_eq!(format!("{history:?}"), format!("{before:?}"));
    }

    #[test]
    fn old_exchanges_are_condensed_and_the_task_and_recent_work_survive() {
        let big = format!("START {} END", "x".repeat(20_000));
        let mut history = vec![
            ToolTurn::User("earlier chat".into()),
            ToolTurn::Assistant("earlier reply".into()),
            ToolTurn::User("Find why the deploy failed.".into()),
        ];
        for n in 0..6 {
            history.extend(exchange(n, &big));
        }
        let report = HistoryCompactor::new(16_000).compact(&mut history, Some(2));
        assert!(report.changed());
        assert!(report.tokens_after <= 16_000, "{report:?}");
        let task = report.task_index.unwrap();
        assert!(matches!(&history[task], ToolTurn::User(t) if t == "Find why the deploy failed."));
        assert!(results_follow_calls(&history), "{history:?}");
        // Three recent exchanges stay, with their outputs' ends visible.
        assert_eq!(exchange_starts(&history).len(), 3);
        // Condensed evidence keeps both ends of older outputs.
        let note = history
            .iter()
            .find_map(|turn| match turn {
                ToolTurn::Assistant(text) if text.starts_with("[Earlier tool results") => {
                    Some(text)
                }
                _ => None,
            })
            .expect("a condensed note");
        assert!(
            note.contains("read_log") && note.contains("START") && note.contains("END"),
            "{note}"
        );
        assert_eq!(report.folded_exchanges, 3);
    }

    #[test]
    fn digesting_never_removes_the_task_even_when_it_is_old() {
        let mut history = vec![ToolTurn::User(
            "Rotate the TLS certificate on edge-2.".into(),
        )];
        for n in 0..4 {
            history.push(ToolTurn::Assistant(format!(
                "thinking {n} {}",
                "y".repeat(8_000)
            )));
            history.extend(exchange(n, &"z".repeat(8_000)));
        }
        let report = HistoryCompactor::new(3_000)
            .keep_recent_exchanges(1)
            .compact(&mut history, Some(0));
        assert_eq!(report.task_index, Some(0));
        assert!(matches!(&history[0], ToolTurn::User(t) if t.starts_with("Rotate the TLS")));
        assert!(report.digested_turns > 0 || report.folded_exchanges > 0);
        assert!(results_follow_calls(&history), "{history:?}");
        assert!(matches!(history.last(), Some(ToolTurn::ToolResults(_))));
    }

    #[test]
    fn earlier_conversation_before_the_task_is_digested_not_dropped() {
        let mut history = Vec::new();
        for n in 0..10 {
            history.push(ToolTurn::User(format!(
                "old question {n} {}",
                "q".repeat(4_000)
            )));
            history.push(ToolTurn::Assistant(format!("old answer {n}")));
        }
        history.push(ToolTurn::User("Now restart the worker.".into()));
        let task = history.len() - 1;
        let report = HistoryCompactor::new(2_000).compact(&mut history, Some(task));
        let task = report.task_index.unwrap();
        assert!(matches!(&history[task], ToolTurn::User(t) if t == "Now restart the worker."));
        let ToolTurn::User(note) = &history[0] else {
            panic!("the digest leads as a user note: {history:?}");
        };
        assert!(note.starts_with("[Earlier conversation, condensed]"));
        assert!(note.contains("old question 0") && note.contains("old answer 9"));
    }

    #[test]
    fn repeated_compaction_merges_notes_and_keeps_result_envelopes_readable() {
        let envelope = |n: usize| {
            crate::ToolResult::<()>::observed(format!(
                "FIRST {n}\n{}LAST {n}",
                "noise\n".repeat(3_000)
            ))
            .for_model()
        };
        let compactor = HistoryCompactor::new(2_500).keep_recent_exchanges(1);
        let mut history = vec![ToolTurn::User("Find the failure.".into())];
        let mut task = Some(0);
        for n in 0..5 {
            history.extend(exchange(n, &envelope(n)));
            task = compactor.compact(&mut history, task).task_index;
        }
        assert!(matches!(&history[task.unwrap()], ToolTurn::User(t) if t == "Find the failure."));
        let notes = history
            .iter()
            .filter_map(|turn| note_body(turn).map(|_| turn))
            .collect::<Vec<_>>();
        assert!(!notes.is_empty());
        for note in &notes {
            let (ToolTurn::Assistant(text) | ToolTurn::User(text)) = note else {
                unreachable!()
            };
            assert_eq!(text.matches(NOTE_PREFIX).count(), 1, "nested note: {text}");
            assert!(
                !text.contains("continuation"),
                "envelope boilerplate kept: {text}"
            );
        }
        let all = format!("{history:?}");
        for n in 0..4 {
            assert!(
                all.contains(&format!("FIRST {n}")),
                "evidence for call {n} lost"
            );
        }
        // Capped envelopes stay valid JSON.
        for turn in &history {
            if let ToolTurn::ToolResults(results) = turn {
                for (_, output) in results {
                    assert!(serde_json::from_str::<serde_json::Value>(output).is_ok());
                }
            }
        }
    }

    #[tokio::test]
    async fn a_model_can_write_the_condensed_notes() {
        let mut model = ScriptedModel::new();
        model.push_text("read_log showed the disk filled at 03:12.");
        let mut history = vec![ToolTurn::User("Why did the job fail?".into())];
        for n in 0..5 {
            history.extend(exchange(n, &"w".repeat(10_000)));
        }
        let report = HistoryCompactor::new(8_000)
            .compact_with_model(&model, "caller", &mut history, Some(0))
            .await;
        assert!(report.folded_exchanges > 0);
        let note = history
            .iter()
            .find_map(|turn| match turn {
                ToolTurn::Assistant(text) if text.starts_with("[Earlier tool results") => {
                    Some(text)
                }
                _ => None,
            })
            .unwrap();
        assert!(note.contains("disk filled at 03:12"), "{note}");
    }
}
