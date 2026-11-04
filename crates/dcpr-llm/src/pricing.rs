//! Model prices, for providers that do not report each call's cost.
//!
//! OpenRouter reports a cost on every call; OpenAI, Anthropic and Gemini do
//! not. A [`PriceTable`] prices a [`Usage`] from list prices, and
//! [`PricedUsage`] fills in that cost before usage reaches your sink.
//!
//! The built-in rows are list prices from OpenRouter's catalogue
//! (`/api/v1/models`) on 2026-11-04, which match the upstream providers' own
//! list prices. Prices change: refresh them with
//! [`PriceTable::from_openrouter_catalogue`] or override rows with
//! [`PriceTable::with_price`]. Batch discounts are not applied.

use crate::{Usage, UsageSink};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// A model's list prices in US cents per million tokens, and its context
/// window when known.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelPrice {
    pub input_cents_per_mtok: f64,
    pub output_cents_per_mtok: f64,
    /// Cache reads. `None` charges them at the input rate.
    pub cache_read_cents_per_mtok: Option<f64>,
    /// Cache writes. `None` charges them at the input rate.
    pub cache_write_cents_per_mtok: Option<f64>,
    pub context_tokens: Option<u64>,
}

impl ModelPrice {
    pub const fn new(input_cents_per_mtok: f64, output_cents_per_mtok: f64) -> Self {
        Self {
            input_cents_per_mtok,
            output_cents_per_mtok,
            cache_read_cents_per_mtok: None,
            cache_write_cents_per_mtok: None,
            context_tokens: None,
        }
    }

    pub const fn with_cache(mut self, read: f64, write: Option<f64>) -> Self {
        self.cache_read_cents_per_mtok = Some(read);
        self.cache_write_cents_per_mtok = write;
        self
    }

    pub const fn with_context(mut self, tokens: u64) -> Self {
        self.context_tokens = Some(tokens);
        self
    }

    /// The cost of `usage` in US cents. Cache-write tokens are part of
    /// `input_tokens` (see [`Usage`]) and are charged at the write rate;
    /// cache reads are separate. Output includes any reasoning tokens.
    pub fn cost_cents(&self, usage: &Usage) -> f64 {
        let per_token = |cents_per_mtok: f64| cents_per_mtok / 1_000_000.0;
        let writes = usage.cache_write_tokens.min(usage.input_tokens);
        let plain_input = usage.input_tokens - writes;
        plain_input as f64 * per_token(self.input_cents_per_mtok)
            + writes as f64
                * per_token(
                    self.cache_write_cents_per_mtok
                        .unwrap_or(self.input_cents_per_mtok),
                )
            + usage.cache_read_tokens as f64
                * per_token(
                    self.cache_read_cents_per_mtok
                        .unwrap_or(self.input_cents_per_mtok),
                )
            + usage.output_tokens as f64 * per_token(self.output_cents_per_mtok)
    }
}

/// Model ids to prices. Lookups ignore a routing prefix (`openai/`,
/// `~anthropic/`), a `:variant` suffix and a date suffix, and treat
/// `claude-sonnet-5-5` and `claude-sonnet-5.5` alike. Otherwise matches are
/// exact: a `-pro` or `-mini` model never takes its sibling's price.
#[derive(Debug, Clone, Default)]
pub struct PriceTable {
    rows: HashMap<String, ModelPrice>,
}

/// One built-in row: (model, input, output, cache read, cache write, context),
/// prices in cents per million tokens.
type Row = (&'static str, f64, f64, f64, Option<f64>, u64);

const BUILT_IN: &[Row] = &[
    // OpenAI
    ("gpt-6.1-sol", 200.0, 1000.0, 10.0, Some(250.0), 1_050_000),
    (
        "gpt-6.1-sol-pro",
        200.0,
        1000.0,
        10.0,
        Some(250.0),
        1_050_000,
    ),
    ("gpt-6.1-terra", 100.0, 500.0, 5.0, Some(125.0), 1_050_000),
    (
        "gpt-6-astra",
        1000.0,
        5000.0,
        100.0,
        Some(1250.0),
        1_050_000,
    ),
    ("gpt-6-sol", 200.0, 1000.0, 20.0, Some(250.0), 1_050_000),
    ("gpt-6-luna", 10.0, 50.0, 1.0, Some(12.5), 1_050_000),
    ("gpt-5.6-sol", 400.0, 2000.0, 40.0, Some(500.0), 1_050_000),
    ("gpt-5.6-terra", 200.0, 1200.0, 20.0, Some(250.0), 1_050_000),
    ("gpt-5.6-luna", 20.0, 120.0, 2.0, Some(25.0), 1_050_000),
    ("gpt-5.5", 500.0, 3000.0, 50.0, None, 1_050_000),
    ("gpt-5.5-pro", 3000.0, 18000.0, 3000.0, None, 1_050_000),
    ("gpt-5.4", 250.0, 1500.0, 25.0, None, 1_050_000),
    ("gpt-5.4-mini", 75.0, 450.0, 7.5, None, 400_000),
    ("gpt-5.4-nano", 20.0, 125.0, 2.0, None, 400_000),
    ("gpt-5", 125.0, 1000.0, 12.5, None, 400_000),
    ("gpt-5-mini", 25.0, 200.0, 2.5, None, 400_000),
    ("gpt-5-nano", 5.0, 40.0, 0.5, None, 400_000),
    ("gpt-4.1", 200.0, 800.0, 50.0, None, 1_047_576),
    ("gpt-4.1-mini", 40.0, 160.0, 10.0, None, 1_047_576),
    ("gpt-4.1-nano", 10.0, 40.0, 3.0, None, 1_047_576),
    ("gpt-4o", 250.0, 1000.0, 125.0, None, 128_000),
    ("gpt-4o-mini", 15.0, 60.0, 7.5, None, 128_000),
    ("o3", 200.0, 800.0, 50.0, None, 200_000),
    // Anthropic
    (
        "claude-fable-5.1",
        1000.0,
        5000.0,
        25.0,
        Some(1250.0),
        1_000_000,
    ),
    (
        "claude-fable-5",
        1000.0,
        5000.0,
        100.0,
        Some(1250.0),
        1_000_000,
    ),
    (
        "claude-opus-5.5",
        400.0,
        2000.0,
        20.0,
        Some(500.0),
        1_000_000,
    ),
    ("claude-opus-5", 500.0, 2500.0, 50.0, Some(625.0), 1_000_000),
    (
        "claude-sonnet-5.5",
        200.0,
        1000.0,
        10.0,
        Some(250.0),
        1_000_000,
    ),
    (
        "claude-sonnet-5",
        200.0,
        1000.0,
        20.0,
        Some(250.0),
        1_000_000,
    ),
    ("claude-haiku-5.5", 10.0, 50.0, 1.0, Some(12.5), 1_000_000),
    (
        "claude-opus-4.8",
        500.0,
        2500.0,
        50.0,
        Some(625.0),
        1_000_000,
    ),
    (
        "claude-opus-4.7",
        500.0,
        2500.0,
        50.0,
        Some(625.0),
        1_000_000,
    ),
    (
        "claude-opus-4.6",
        500.0,
        2500.0,
        50.0,
        Some(625.0),
        1_000_000,
    ),
    ("claude-opus-4.5", 500.0, 2500.0, 50.0, Some(625.0), 200_000),
    (
        "claude-sonnet-4.6",
        300.0,
        1500.0,
        30.0,
        Some(375.0),
        1_000_000,
    ),
    (
        "claude-sonnet-4.5",
        300.0,
        1500.0,
        30.0,
        Some(375.0),
        1_000_000,
    ),
    ("claude-haiku-4.5", 100.0, 500.0, 10.0, Some(125.0), 200_000),
    // Google
    ("gemini-3.8-flash", 82.5, 412.5, 8.25, None, 1_048_576),
    ("gemini-3.7-flash", 75.0, 375.0, 7.5, None, 1_048_576),
    ("gemini-3.6-flash", 75.0, 375.0, 7.5, None, 1_048_576),
    ("gemini-3.5-flash", 150.0, 900.0, 15.0, None, 1_048_576),
    ("gemini-3.5-flash-lite", 30.0, 250.0, 3.0, None, 1_048_576),
    ("gemini-3.1-flash-lite", 25.0, 150.0, 2.5, None, 1_048_576),
    (
        "gemini-3.1-pro-preview",
        200.0,
        1200.0,
        20.0,
        None,
        1_048_576,
    ),
    ("gemini-3-flash-preview", 50.0, 300.0, 5.0, None, 1_048_576),
];

/// The lookup key for a model id. See [`PriceTable`].
fn normalise(model: &str) -> String {
    let name = model.trim().trim_start_matches('~');
    let name = name.rsplit('/').next().unwrap_or(name);
    let name = name.split(':').next().unwrap_or(name);
    // A date suffix: -2026-04-23 or -20251101.
    let parts: Vec<&str> = name.split('-').collect();
    let mut end = parts.len();
    let is_digits =
        |part: &str, len: usize| part.len() == len && part.bytes().all(|b| b.is_ascii_digit());
    if end >= 4
        && is_digits(parts[end - 3], 4)
        && is_digits(parts[end - 2], 2)
        && is_digits(parts[end - 1], 2)
    {
        end -= 3;
    } else if end >= 2 && is_digits(parts[end - 1], 8) {
        end -= 1;
    }
    let mut name = parts[..end].join("-").to_ascii_lowercase();
    if name.starts_with("claude-") {
        // claude-sonnet-5-5 → claude-sonnet-5.5
        let bytes: Vec<char> = name.chars().collect();
        name = bytes
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let between_digits = *c == '-'
                    && i > 0
                    && bytes[i - 1].is_ascii_digit()
                    && bytes.get(i + 1).is_some_and(char::is_ascii_digit);
                if between_digits { '.' } else { *c }
            })
            .collect();
    }
    name
}

impl PriceTable {
    /// An empty table.
    pub fn empty() -> Self {
        Self::default()
    }

    /// The built-in list prices. See the module documentation for their date.
    pub fn built_in() -> Self {
        let mut table = Self::empty();
        for &(model, input, output, read, write, context) in BUILT_IN {
            table = table.with_price(
                model,
                ModelPrice::new(input, output)
                    .with_cache(read, write)
                    .with_context(context),
            );
        }
        table
    }

    /// Add or replace one model's price.
    pub fn with_price(mut self, model: &str, price: ModelPrice) -> Self {
        self.rows.insert(normalise(model), price);
        self
    }

    /// Rows from an OpenRouter model catalogue (the JSON of
    /// `GET https://openrouter.ai/api/v1/models`), which lists per-token USD
    /// prices. `:batch` and other variants are skipped. Merge it over the
    /// built-in rows with [`PriceTable::merge`].
    pub fn from_openrouter_catalogue(catalogue: &serde_json::Value) -> Self {
        let mut table = Self::empty();
        for model in catalogue["data"].as_array().into_iter().flatten() {
            let Some(id) = model["id"].as_str().filter(|id| !id.contains(':')) else {
                continue;
            };
            let pricing = &model["pricing"];
            let cents = |key: &str| {
                pricing[key]
                    .as_str()
                    .and_then(|value| value.parse::<f64>().ok())
                    .map(|usd_per_token| usd_per_token * 100.0 * 1_000_000.0)
            };
            let (Some(input), Some(output)) = (cents("prompt"), cents("completion")) else {
                continue;
            };
            let mut price = ModelPrice::new(input, output);
            price.cache_read_cents_per_mtok = cents("input_cache_read").filter(|c| *c > 0.0);
            price.cache_write_cents_per_mtok = cents("input_cache_write").filter(|c| *c > 0.0);
            price.context_tokens = model["context_length"].as_u64();
            table = table.with_price(id, price);
        }
        table
    }

    /// This table with `other`'s rows added, replacing any it shares.
    pub fn merge(mut self, other: PriceTable) -> Self {
        self.rows.extend(other.rows);
        self
    }

    pub fn price(&self, model: &str) -> Option<ModelPrice> {
        self.rows.get(&normalise(model)).copied()
    }

    /// The cost of `usage` on `model` in US cents, or `None` for a model
    /// without a row.
    pub fn cost_cents(&self, model: &str, usage: &Usage) -> Option<f64> {
        self.price(model).map(|price| price.cost_cents(usage))
    }

    /// The model's context window in tokens, when known.
    pub fn context_tokens(&self, model: &str) -> Option<u64> {
        self.price(model).and_then(|price| price.context_tokens)
    }
}

/// A [`UsageSink`] that prices each call the provider did not price, then
/// passes it on. A model missing from the table is passed on unpriced and
/// warned about once.
pub struct PricedUsage<C: ?Sized> {
    inner: Arc<dyn UsageSink<C>>,
    table: PriceTable,
    warned: Mutex<HashSet<String>>,
}

impl<C: ?Sized> PricedUsage<C> {
    pub fn new(inner: Arc<dyn UsageSink<C>>, table: PriceTable) -> Self {
        Self {
            inner,
            table,
            warned: Mutex::new(HashSet::new()),
        }
    }
}

impl<C: ?Sized + Sync> UsageSink<C> for PricedUsage<C> {
    fn record(&self, caller: &C, model: &str, usage: Usage, reported_cost_cents: Option<f64>) {
        let cost = reported_cost_cents.or_else(|| {
            let priced = self.table.cost_cents(model, &usage);
            if priced.is_none()
                && self
                    .warned
                    .lock()
                    .is_ok_and(|mut warned| warned.insert(model.to_string()))
            {
                tracing::warn!(
                    model,
                    "no price for this model; its calls are recorded without a cost"
                );
            }
            priced
        });
        self.inner.record(caller, model, usage, cost);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn ids_match_across_routing_prefixes_variants_dates_and_claude_spellings() {
        let table = PriceTable::built_in();
        let sol = table.price("gpt-6.1-sol").unwrap();
        for id in [
            "openai/gpt-6.1-sol",
            "gpt-6.1-sol:nitro",
            "gpt-6.1-sol-2026-09-30",
        ] {
            assert_eq!(table.price(id), Some(sol), "{id}");
        }
        let sonnet = table.price("anthropic/claude-sonnet-5.5").unwrap();
        assert_eq!(table.price("claude-sonnet-5-5"), Some(sonnet));
        assert_eq!(table.price("~anthropic/claude-sonnet-5.5"), Some(sonnet));
        assert!(table.price("claude-opus-4-5-20251101").is_some());
        // Siblings never share a price.
        assert_ne!(table.price("gpt-5.5"), table.price("gpt-5.5-pro"));
        assert!(table.price("gpt-5.5-turbo").is_none());
        assert!(table.price("unknown/model").is_none());
    }

    #[test]
    fn cache_reads_and_writes_use_their_own_rates() {
        let price = ModelPrice::new(200.0, 1000.0).with_cache(10.0, Some(250.0));
        let usage = Usage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            cache_read_tokens: 1_000_000,
            cache_write_tokens: 400_000,
        };
        // 600k plain input, 400k writes, 1M reads, 1M output.
        let expected = 0.6 * 200.0 + 0.4 * 250.0 + 10.0 + 1000.0;
        assert!((price.cost_cents(&usage) - expected).abs() < 1e-9);
        // Without cache rates every input token is charged at the input rate.
        let plain = ModelPrice::new(200.0, 1000.0);
        assert!((plain.cost_cents(&usage) - (2.0 * 200.0 + 1000.0)).abs() < 1e-9);
    }

    #[test]
    fn an_openrouter_catalogue_becomes_rows_in_cents_per_mtok() {
        let catalogue = json!({"data": [
            {"id": "openai/gpt-7", "context_length": 2_000_000, "pricing": {
                "prompt": "0.000002", "completion": "0.00001", "input_cache_read": "0.0000001"}},
            {"id": "openai/gpt-7:batch", "pricing": {"prompt": "0.000001", "completion": "0.000005"}},
        ]});
        let table = PriceTable::built_in().merge(PriceTable::from_openrouter_catalogue(&catalogue));
        let price = table.price("gpt-7").unwrap();
        assert!((price.input_cents_per_mtok - 200.0).abs() < 1e-6);
        assert!((price.output_cents_per_mtok - 1000.0).abs() < 1e-6);
        assert!((price.cache_read_cents_per_mtok.unwrap() - 10.0).abs() < 1e-6);
        assert_eq!(price.cache_write_cents_per_mtok, None);
        assert_eq!(table.context_tokens("gpt-7"), Some(2_000_000));
        assert!(
            table.price("gpt-6.1-sol").is_some(),
            "built-in rows survive the merge"
        );
    }

    #[derive(Default)]
    struct Seen(Mutex<Vec<Option<f64>>>);
    impl UsageSink<str> for Seen {
        fn record(&self, _: &str, _: &str, _: Usage, cost: Option<f64>) {
            self.0.lock().unwrap().push(cost);
        }
    }

    #[test]
    fn priced_usage_fills_only_missing_costs() {
        let seen = Arc::new(Seen::default());
        let sink = PricedUsage::new(seen.clone(), PriceTable::built_in());
        let usage = Usage {
            input_tokens: 1_000_000,
            ..Usage::default()
        };
        sink.record("caller", "gpt-6.1-sol", usage, None);
        sink.record("caller", "openai/gpt-6.1-sol", usage, Some(7.0));
        sink.record("caller", "unknown/model", usage, None);
        assert_eq!(*seen.0.lock().unwrap(), vec![Some(200.0), Some(7.0), None]);
    }
}
