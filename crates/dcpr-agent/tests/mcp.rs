//! A tool served by `ToolServer` and called back through `McpTool`, over an
//! in-memory rmcp transport.

use async_trait::async_trait;
use dcpr_agent::mcp::{McpTool, ToolServer};
use dcpr_agent::{Runtime, Tool, ToolOutcomeStatus, ToolResult};
use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParams;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

struct Greenhouse;

impl Runtime for Greenhouse {
    type Context = Mutex<Vec<String>>;
    type Error = String;
    type Extra = ();
}

struct WaterBed;

#[async_trait]
impl Tool<Greenhouse> for WaterBed {
    fn name(&self) -> &str {
        "water_bed"
    }
    fn description(&self) -> &str {
        "Water one raised bed."
    }
    fn parameters(&self) -> Value {
        json!({"type":"object","properties":{"bed":{"type":"integer"}},"required":["bed"]})
    }
    fn is_action(&self) -> bool {
        true
    }
    async fn call(&self, log: &Mutex<Vec<String>>, args: Value) -> Result<ToolResult, String> {
        let bed = args["bed"].as_i64().unwrap();
        if bed > 9 {
            return Err(format!("there is no bed {bed}"));
        }
        log.lock().unwrap().push(format!("watered {bed}"));
        Ok(ToolResult::text(format!("Bed {bed} watered.")).with_evidence(json!({"bed": bed})))
    }
}

struct ReadThermometer;

#[async_trait]
impl Tool<Greenhouse> for ReadThermometer {
    fn name(&self) -> &str {
        "read_thermometer"
    }
    fn description(&self) -> &str {
        "Read the greenhouse temperature."
    }
    fn parameters(&self) -> Value {
        json!({"type":"object","properties":{}})
    }
    async fn call(&self, _log: &Mutex<Vec<String>>, _args: Value) -> Result<ToolResult, String> {
        Ok(ToolResult::observed("21 C"))
    }
}

/// A client-side runtime unrelated to the server's.
struct Gardener;

impl Runtime for Gardener {
    type Context = ();
    type Error = String;
    type Extra = ();
}

async fn connect() -> (
    Arc<Mutex<Vec<String>>>,
    rmcp::service::RunningService<rmcp::RoleClient, ()>,
) {
    let log = Arc::new(Mutex::new(Vec::new()));
    let server = ToolServer::<Greenhouse>::with_shared_context(
        vec![Box::new(WaterBed), Box::new(ReadThermometer)],
        log.clone(),
    )
    .with_instructions("Tools for one greenhouse.");
    let (server_io, client_io) = tokio::io::duplex(64 * 1024);
    tokio::spawn(async move {
        let running = server.serve(server_io).await.expect("server starts");
        let _ = running.waiting().await;
    });
    let client = ().serve(client_io).await.expect("client connects");
    (log, client)
}

#[tokio::test]
async fn served_tools_round_trip_through_mcp() {
    let (log, client) = connect().await;
    assert_eq!(
        client
            .peer_info()
            .and_then(|info| info.instructions.clone())
            .as_deref(),
        Some("Tools for one greenhouse.")
    );

    let tools: Vec<Box<dyn Tool<Gardener>>> = McpTool::list(&client)
        .await
        .unwrap()
        .into_iter()
        .map(|tool| tool.with_prefix("greenhouse").boxed())
        .collect();
    let names: Vec<&str> = tools.iter().map(|tool| tool.name()).collect();
    assert_eq!(
        names,
        ["greenhouse__water_bed", "greenhouse__read_thermometer"]
    );
    let water = &tools[0];
    let thermometer = &tools[1];
    assert!(water.is_action());
    assert!(!thermometer.is_action());
    assert_eq!(water.parameters()["required"], json!(["bed"]));

    let watered = water.call(&(), json!({"bed": 3})).await.unwrap();
    assert_eq!(watered.status, ToolOutcomeStatus::Applied);
    assert_eq!(watered.output, "Bed 3 watered.");
    assert!(watered.is_untrusted());
    assert_eq!(
        watered.evidence["structured_content"]["evidence"],
        json!({"bed": 3})
    );
    assert_eq!(*log.lock().unwrap(), ["watered 3"]);

    let reading = thermometer.call(&(), json!({})).await.unwrap();
    let reading = reading.with_runtime_semantics(thermometer.is_action());
    assert_eq!(reading.status, ToolOutcomeStatus::Observed);
    assert_eq!(reading.output, "21 C");
}

#[tokio::test]
async fn failures_reach_the_model_as_failed_results() {
    let (log, client) = connect().await;
    let water = McpTool::list(&client)
        .await
        .unwrap()
        .into_iter()
        .find(|tool| tool.remote_name() == "water_bed")
        .unwrap()
        .trusted();

    let missing = Tool::<Gardener>::call(&water, &(), json!({}))
        .await
        .unwrap();
    assert_eq!(missing.status, ToolOutcomeStatus::Failed);
    assert!(
        missing.output.contains("Invalid arguments for water_bed"),
        "{}",
        missing.output
    );

    let error = Tool::<Gardener>::call(&water, &(), json!({"bed": 12}))
        .await
        .unwrap();
    assert_eq!(error.status, ToolOutcomeStatus::Failed);
    assert!(
        error.output.contains("there is no bed 12"),
        "{}",
        error.output
    );
    assert!(!error.is_untrusted());
    assert!(log.lock().unwrap().is_empty());
}

#[tokio::test]
async fn an_unknown_tool_is_a_protocol_error() {
    let (_log, client) = connect().await;
    let error = client
        .call_tool(CallToolRequestParams::new("prune_roses"))
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("unknown tool: prune_roses"),
        "{error}"
    );
}
