//! DyTopo CLI - Dynamic topology multi-agent coordination demo.

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use dytopo_agents::{specialties, Agent, LlmWorker, LlmWorkerConfig, StubWorker};
use dytopo_core::AgentId;
use dytopo_embed::HashEmbedder;
use dytopo_llm::{LlmClient, OllamaHost, OllamaPool};
use dytopo_orchestrator::{run_stub, OrchestratorConfig};
use dytopo_router::RouterConfig;
use dytopo_viz::write_dot;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "dytopo-cli")]
#[command(about = "DyTopo-inspired dynamic topology demo (Rust-first)")]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq)]
enum LlmProvider {
    /// No LLM - use stub agents
    None,
    /// Use Ollama for LLM inference
    Ollama,
}

#[derive(Subcommand)]
enum Command {
    /// Run a demo with configurable agents
    Demo {
        /// Number of rounds
        #[arg(long, default_value_t = 3)]
        rounds: usize,

        /// Number of worker agents
        #[arg(long, default_value_t = 5)]
        agents: usize,

        /// Top-K incoming edges per receiver
        #[arg(long, default_value_t = 2)]
        topk: usize,

        /// Minimum similarity score for an edge
        #[arg(long, default_value_t = 0.10)]
        min_score: f32,

        /// Keep at least one incoming edge even if below threshold
        #[arg(long, default_value_t = true)]
        force_connect: bool,

        /// Max inbox messages kept per agent
        #[arg(long, default_value_t = 3)]
        max_inbox: usize,

        /// Task description (manager uses this as context)
        #[arg(long, default_value = "Solve a toy problem by coordinating specialists.")]
        task: String,

        /// Output directory for traces and DOT files
        #[arg(long, default_value = "traces")]
        out: String,

        /// LLM provider to use (none = stub agents)
        #[arg(long, value_enum, default_value_t = LlmProvider::None)]
        llm: LlmProvider,

        /// Model name for LLM (e.g., llama2, mistral, codellama)
        #[arg(long, default_value = "llama2")]
        model: String,

        /// Ollama host URLs (comma-separated, format: name=url:concurrent)
        /// Example: manager=http://manager:11434:2,curiosity=http://curiosity:11434:3
        #[arg(long, default_value = "manager=http://manager.local:11434:2,curiosity=http://curiosity.local:11434:3")]
        ollama_hosts: String,

        /// Stagger delay in milliseconds between requests to same host
        #[arg(long, default_value_t = 500)]
        stagger_ms: u64,
    },
}

/// Parse host specification: name=url:concurrent
fn parse_host(spec: &str) -> Result<OllamaHost> {
    let parts: Vec<&str> = spec.split('=').collect();
    if parts.len() != 2 {
        anyhow::bail!("Invalid host spec '{spec}', expected name=url:concurrent");
    }

    let name = parts[0];
    let url_concurrent: Vec<&str> = parts[1].rsplitn(2, ':').collect();

    if url_concurrent.len() != 2 {
        anyhow::bail!("Invalid host spec '{spec}', expected name=url:concurrent");
    }

    let concurrent: usize = url_concurrent[0].parse()?;
    let url = url_concurrent[1];

    // Handle case where URL has port (http://host:11434:2)
    // rsplitn splits from the right, so url_concurrent[1] is everything before last :
    // and url_concurrent[0] is the concurrent count

    Ok(OllamaHost::new(name, url, concurrent))
}

/// Parse comma-separated host specifications.
fn parse_hosts(specs: &str) -> Result<Vec<OllamaHost>> {
    specs.split(',').map(|s| parse_host(s.trim())).collect()
}

/// Get specialty for agent based on index (rotating through available specialties).
fn get_specialty(index: usize) -> &'static str {
    let specialties = [
        specialties::MATHEMATICIAN,
        specialties::CODER,
        specialties::PLANNER,
        specialties::WRITER,
        specialties::REVIEWER,
        specialties::RESEARCHER,
    ];
    specialties[index % specialties.len()]
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.cmd {
        Command::Demo {
            rounds,
            agents,
            topk,
            min_score,
            force_connect,
            max_inbox,
            task,
            out,
            llm,
            model,
            ollama_hosts,
            stagger_ms,
        } => {
            let embedder = HashEmbedder::new(128, 42);

            let mut cfg = OrchestratorConfig::default();
            cfg.rounds = rounds;
            cfg.max_inbox = max_inbox;

            let mut r = RouterConfig::default();
            r.topk_per_receiver = topk;
            r.min_score = min_score;
            r.force_connect = force_connect;
            cfg.router = r;

            // Create agents based on LLM provider
            let workers: Vec<Box<dyn Agent>> = match llm {
                LlmProvider::None => {
                    println!("Using stub agents (no LLM)");
                    (0..agents)
                        .map(|i| {
                            let id = AgentId(i + 1);
                            Box::new(StubWorker::new(id, 1234)) as Box<dyn Agent>
                        })
                        .collect()
                }
                LlmProvider::Ollama => {
                    let hosts = parse_hosts(&ollama_hosts)?;

                    println!("Using Ollama LLM with {} hosts:", hosts.len());
                    for host in &hosts {
                        println!(
                            "  {} @ {} (max {} concurrent)",
                            host.name, host.base_url, host.max_concurrent
                        );
                    }
                    println!("Model: {model}");
                    println!("Stagger delay: {stagger_ms}ms");

                    let pool = Arc::new(
                        OllamaPool::new(hosts, &model)
                            .with_stagger_delay(std::time::Duration::from_millis(stagger_ms)),
                    ) as Arc<dyn LlmClient>;

                    (0..agents)
                        .map(|i| {
                            let id = AgentId(i + 1);
                            let config =
                                LlmWorkerConfig::default().with_specialty(get_specialty(i));
                            Box::new(LlmWorker::new(id, Arc::clone(&pool), config))
                                as Box<dyn Agent>
                        })
                        .collect()
                }
            };

            println!("Starting demo with {} agents, {} rounds", agents, rounds);
            println!("Task: {task}");
            println!();

            let run = run_stub(&task, workers, &embedder, &cfg, &out, "demo")?;
            println!("\nTrace written to: {}", run.trace_path);

            for topo in &run.topologies {
                let dot = format!("{out}/topology_demo_round{}.dot", topo.round);
                write_dot(&dot, topo)?;
                println!("\nWrote DOT: {dot}");
                println!("Round {} edges:", topo.round);
                for e in &topo.edges {
                    println!("  {} -> {}  ({:.3})", e.from.0, e.to.0, e.score);
                }
            }

            // Print load stats if using Ollama
            if llm == LlmProvider::Ollama {
                println!("\nDemo complete. Check traces for full output.");
            }
        }
    }

    Ok(())
}
