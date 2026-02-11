//! DyTopo CLI - Dynamic topology multi-agent coordination demo.

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use dytopo_agents::{specialties, Agent, LlmWorker, LlmWorkerConfig, StubWorker};
use dytopo_analyze::{
    compute_metrics, compute_total_tokens, evaluate_quality, extract_solution,
    format_report, generate_all_plots, load_trace, plot_quality_comparison, QualityEval,
};
use dytopo_core::AgentId;
use dytopo_embed::{Embedder, HashEmbedder, OllamaEmbedder};
use dytopo_llm::{LlmClient, OllamaHost, OllamaPool};
use dytopo_orchestrator::{run_stub, OrchestratorConfig};
use dytopo_router::{BaselineTopology, RouterConfig};
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

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq)]
enum EmbedderType {
    /// Hash-based embeddings (deterministic, not semantic)
    Hash,
    /// Ollama embeddings (semantic, using nomic-embed-text or similar)
    Ollama,
}

#[derive(Subcommand)]
enum Command {
    /// Analyze a trace file and generate metrics/plots
    Analyze {
        /// Path to the JSONL trace file
        #[arg(long)]
        trace: String,

        /// Output directory for plots (default: same as trace)
        #[arg(long)]
        out: Option<String>,

        /// Output format (text, json, csv)
        #[arg(long, default_value = "text")]
        format: String,

        /// Skip plot generation
        #[arg(long)]
        no_plots: bool,
    },

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

        /// Model name for LLM (e.g., llama2, mistral, qwen2.5:7b-instruct)
        #[arg(long, default_value = "llama2")]
        model: String,

        /// Ollama host URLs for LLM (comma-separated, format: name=url:concurrent)
        /// Example: manager=http://manager:11434:2,curiosity=http://curiosity:11434:3
        #[arg(long, default_value = "manager=http://manager.local:11434:2,curiosity=http://curiosity.local:11434:3")]
        ollama_hosts: String,

        /// Stagger delay in milliseconds between LLM requests to same host
        #[arg(long, default_value_t = 500)]
        stagger_ms: u64,

        /// Embedder type for routing (hash = deterministic, ollama = semantic)
        #[arg(long, value_enum, default_value_t = EmbedderType::Hash)]
        embedder: EmbedderType,

        /// Embedding model name (for ollama embedder)
        #[arg(long, default_value = "nomic-embed-text")]
        embed_model: String,

        /// Ollama URL for embeddings (defaults to first LLM host)
        #[arg(long)]
        embed_url: Option<String>,
    },

    /// Compare dynamic routing against baseline topologies
    Benchmark {
        /// Task description
        #[arg(long, default_value = "Write a Python function to sort a list")]
        task: String,

        /// Number of agents
        #[arg(long, default_value_t = 5)]
        agents: usize,

        /// Number of rounds per experiment
        #[arg(long, default_value_t = 3)]
        rounds: usize,

        /// Output directory
        #[arg(long, default_value = "benchmark")]
        out: String,

        /// Ollama URL for embeddings and LLM judge
        #[arg(long, default_value = "http://localhost:11434")]
        embed_url: String,

        /// Embedding model
        #[arg(long, default_value = "nomic-embed-text")]
        embed_model: String,

        /// LLM model for judge (to evaluate quality)
        #[arg(long, default_value = "qwen2.5:7b-instruct")]
        judge_model: String,

        /// Skip LLM quality evaluation (faster, metrics only)
        #[arg(long)]
        no_eval: bool,
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

    Ok(OllamaHost::new(name, url, concurrent))
}

/// Parse comma-separated host specifications.
fn parse_hosts(specs: &str) -> Result<Vec<OllamaHost>> {
    specs.split(',').map(|s| parse_host(s.trim())).collect()
}

/// Extract base URL from first host spec (for embeddings).
fn extract_first_url(specs: &str) -> Result<String> {
    let host = parse_host(specs.split(',').next().unwrap_or(specs).trim())?;
    Ok(host.base_url)
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
        Command::Analyze {
            trace,
            out,
            format,
            no_plots,
        } => {
            println!("Loading trace: {trace}");
            let parsed = load_trace(&trace)?;
            let metrics = compute_metrics(&parsed, &trace);

            match format.as_str() {
                "json" => {
                    let json = serde_json::to_string_pretty(&metrics)?;
                    println!("{json}");
                }
                "csv" => {
                    println!("round,duration_ms,edges,avg_score,min_score,max_score,density");
                    for r in &metrics.rounds {
                        println!(
                            "{},{},{},{:.4},{:.4},{:.4},{:.4}",
                            r.round,
                            r.duration_ms,
                            r.edge_count,
                            r.avg_score,
                            r.min_score,
                            r.max_score,
                            r.graph_density
                        );
                    }
                }
                _ => {
                    let report = format_report(&metrics);
                    println!("{report}");
                }
            }

            if !no_plots {
                let out_dir = out.unwrap_or_else(|| {
                    std::path::Path::new(&trace)
                        .parent()
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_else(|| ".".to_string())
                });
                let plot_paths = generate_all_plots(&metrics, &out_dir)?;
                println!("\nGenerated plots:");
                for p in plot_paths {
                    println!("  {p}");
                }
            }
        }

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
            embedder,
            embed_model,
            embed_url,
        } => {
            // Create embedder based on type
            let embedder_impl: Box<dyn Embedder> = match embedder {
                EmbedderType::Hash => {
                    println!("Using hash embeddings (deterministic, not semantic)");
                    Box::new(HashEmbedder::new(128, 42))
                }
                EmbedderType::Ollama => {
                    let url = embed_url
                        .clone()
                        .unwrap_or_else(|| extract_first_url(&ollama_hosts).unwrap_or_else(|_| "http://localhost:11434".to_string()));
                    println!("Using Ollama semantic embeddings:");
                    println!("  URL: {url}");
                    println!("  Model: {embed_model}");
                    Box::new(OllamaEmbedder::new(&url, &embed_model)?)
                }
            };

            let cfg = OrchestratorConfig {
                rounds,
                max_inbox,
                router: RouterConfig {
                    topk_per_receiver: topk,
                    min_score,
                    force_connect,
                },
            };

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

            println!("\nStarting demo with {} agents, {} rounds", agents, rounds);
            println!("Task: {task}");
            println!();

            let run = run_stub(&task, workers, embedder_impl.as_ref(), &cfg, &out, "demo")?;
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

            // Print summary
            if llm == LlmProvider::Ollama || embedder == EmbedderType::Ollama {
                println!("\nDemo complete. Check traces for full output.");
            }
        }

        Command::Benchmark {
            task,
            agents,
            rounds,
            out,
            embed_url,
            embed_model,
            judge_model,
            no_eval,
        } => {
            use dytopo_llm::OllamaClient;
            use dytopo_router::build_baseline_topology;
            use std::time::Instant;

            std::fs::create_dir_all(&out)?;

            println!("=== DyTopo Benchmark ===\n");
            println!("Task: {task}");
            println!("Agents: {agents}");
            println!("Rounds: {rounds}");
            println!("Embeddings: {embed_url} ({embed_model})");
            if !no_eval {
                println!("Judge Model: {judge_model}");
            }
            println!();

            let agent_ids: Vec<AgentId> = (1..=agents).map(AgentId).collect();

            // Create embedder for dynamic routing
            let embedder = OllamaEmbedder::new(&embed_url, &embed_model)?;

            // Create judge LLM for quality evaluation
            let judge: Option<OllamaClient> = if no_eval {
                None
            } else {
                Some(OllamaClient::new(
                    OllamaHost::new("judge", &embed_url, 1),
                    &judge_model,
                ))
            };

            // Run dynamic routing experiment
            println!("Running DYNAMIC routing experiment...");
            let dynamic_start = Instant::now();

            let cfg = OrchestratorConfig {
                rounds,
                max_inbox: 3,
                router: RouterConfig {
                    topk_per_receiver: 2,
                    min_score: 0.1,
                    force_connect: true,
                },
            };

            let workers: Vec<Box<dyn Agent>> = (0..agents)
                .map(|i| {
                    let id = AgentId(i + 1);
                    Box::new(StubWorker::new(id, 1234)) as Box<dyn Agent>
                })
                .collect();

            let dynamic_run = run_stub(&task, workers, &embedder, &cfg, &out, "dynamic")?;
            let dynamic_duration = dynamic_start.elapsed();

            // Compute metrics for dynamic
            let dynamic_trace = load_trace(&dynamic_run.trace_path)?;
            let dynamic_metrics = compute_metrics(&dynamic_trace, &dynamic_run.trace_path);

            println!("  Duration: {:?}", dynamic_duration);
            println!("  Avg Score: {:.3}", dynamic_metrics.summary.overall_avg_score);
            println!("  Edge Stability: {:.1}%", dynamic_metrics.summary.edge_stability * 100.0);

            // Extract solution and evaluate quality
            let dynamic_quality: Option<QualityEval> = if let Some(ref judge_llm) = judge {
                if let Some(solution) = extract_solution(&dynamic_trace) {
                    let tokens = compute_total_tokens(&dynamic_trace);
                    println!("  Evaluating solution quality...");
                    match evaluate_quality(&task, &solution, tokens, judge_llm) {
                        Ok(eval) => {
                            println!("  Quality Score: {:.1}/10", eval.score);
                            println!("    Correctness: {:.1}, Completeness: {:.1}, Clarity: {:.1}",
                                eval.correctness, eval.completeness, eval.clarity);
                            Some(eval)
                        }
                        Err(e) => {
                            println!("  Quality evaluation failed: {e}");
                            None
                        }
                    }
                } else {
                    println!("  No solution to evaluate (empty trace)");
                    None
                }
            } else {
                None
            };

            println!();

            // Run baseline experiments (just compute topology metrics, no full orchestration)
            let baselines = [
                ("FullyConnected", BaselineTopology::FullyConnected),
                ("Star", BaselineTopology::Star),
                ("Chain", BaselineTopology::Chain),
                ("Ring", BaselineTopology::Ring),
            ];

            println!("Baseline Topologies (edge counts for comparison):\n");
            println!("| Topology | Edges | Density |");
            println!("|----------|-------|---------|");

            for (name, baseline) in &baselines {
                let topo = build_baseline_topology(0, *baseline, &agent_ids);
                let max_edges = agents * (agents - 1);
                let density = topo.edges.len() as f32 / max_edges as f32;
                println!("| {} | {} | {:.1}% |", name, topo.edges.len(), density * 100.0);
            }

            println!();
            println!("Dynamic Routing Summary:");
            println!("  Total Duration: {}ms", dynamic_metrics.summary.total_duration_ms);
            println!("  Avg Edges/Round: {:.1}", dynamic_metrics.summary.avg_edges_per_round);
            println!("  Avg Score: {:.3}", dynamic_metrics.summary.overall_avg_score);
            println!("  Score Trend: {:+.4}/round", dynamic_metrics.summary.score_trend);
            println!("  Edge Stability: {:.1}%", dynamic_metrics.summary.edge_stability * 100.0);

            // Generate comparison plots
            let mut plot_paths = generate_all_plots(&dynamic_metrics, &out)?;

            // Generate quality plot if we have evaluation
            if let Some(ref eval) = dynamic_quality {
                let quality_data = vec![("Dynamic", eval.score)];
                let quality_plot_path = format!("{}/quality.svg", out);
                if plot_quality_comparison(&quality_data, &quality_plot_path).is_ok() {
                    plot_paths.push(quality_plot_path);
                }
            }

            println!("\nGenerated plots:");
            for p in &plot_paths {
                println!("  {p}");
            }

            // Write comparison report
            let report_path = format!("{}/benchmark_report.md", out);
            let mut report = String::new();
            report.push_str("# DyTopo Benchmark Report\n\n");
            report.push_str(&format!("**Task:** {}\n\n", task));
            report.push_str(&format!("**Agents:** {} | **Rounds:** {}\n\n", agents, rounds));

            // Quality evaluation section
            if let Some(ref eval) = dynamic_quality {
                report.push_str("## Solution Quality (LLM-as-Judge)\n\n");
                report.push_str("| Criterion | Score (1-10) |\n");
                report.push_str("|-----------|-------------|\n");
                report.push_str(&format!("| **Overall** | **{:.1}** |\n", eval.score));
                report.push_str(&format!("| Correctness | {:.1} |\n", eval.correctness));
                report.push_str(&format!("| Completeness | {:.1} |\n", eval.completeness));
                report.push_str(&format!("| Clarity | {:.1} |\n", eval.clarity));
                report.push_str(&format!("\n**Reasoning:** {}\n\n", eval.reasoning));
            }

            report.push_str("## Dynamic Routing Results\n\n");
            report.push_str("| Metric | Value |\n|--------|-------|\n");
            report.push_str(&format!("| Total Duration | {}ms |\n", dynamic_metrics.summary.total_duration_ms));
            report.push_str(&format!("| Avg Edges/Round | {:.1} |\n", dynamic_metrics.summary.avg_edges_per_round));
            report.push_str(&format!("| Avg Score | {:.3} |\n", dynamic_metrics.summary.overall_avg_score));
            report.push_str(&format!("| Score Trend | {:+.4}/round |\n", dynamic_metrics.summary.score_trend));
            report.push_str(&format!("| Edge Stability | {:.1}% |\n", dynamic_metrics.summary.edge_stability * 100.0));

            report.push_str("\n## Baseline Comparison\n\n");
            report.push_str("| Topology | Edges | Density | vs Dynamic |\n");
            report.push_str("|----------|-------|---------|------------|\n");

            let dynamic_edges = dynamic_metrics.summary.avg_edges_per_round;
            for (name, baseline) in &baselines {
                let topo = build_baseline_topology(0, *baseline, &agent_ids);
                let max_edges = agents * (agents - 1);
                let density = topo.edges.len() as f32 / max_edges as f32;
                let ratio = topo.edges.len() as f32 / dynamic_edges;
                report.push_str(&format!(
                    "| {} | {} | {:.1}% | {:.1}x |\n",
                    name, topo.edges.len(), density * 100.0, ratio
                ));
            }

            report.push_str("\n## Key Insight\n\n");
            if let Some(ref eval) = dynamic_quality {
                report.push_str(&format!(
                    "Dynamic routing achieved a **quality score of {:.1}/10** while using ",
                    eval.score
                ));
                report.push_str(&format!(
                    "only **{:.0}% of the edges** that a fully-connected topology would use. ",
                    (dynamic_edges / (agents * (agents - 1)) as f32) * 100.0
                ));
                report.push_str("This demonstrates that semantic matching can achieve good results ");
                report.push_str("with significantly fewer connections.\n\n");
            } else {
                report.push_str("Dynamic routing achieves **semantic matching** between agents, ");
                report.push_str("connecting those with complementary needs/offers. ");
                report.push_str(&format!(
                    "With an average score of {:.3}, agents are matched based on semantic similarity ",
                    dynamic_metrics.summary.overall_avg_score
                ));
                report.push_str("rather than arbitrary topology.\n\n");
            }

            // Solution excerpt
            if let Some(solution) = extract_solution(&dynamic_trace) {
                report.push_str("## Solution Excerpt\n\n");
                report.push_str("```\n");
                // Truncate to first 1000 chars
                let excerpt: String = solution.chars().take(1000).collect();
                report.push_str(&excerpt);
                if solution.len() > 1000 {
                    report.push_str("\n... (truncated)");
                }
                report.push_str("\n```\n\n");
            }

            report.push_str("## Plots\n\n");
            for p in &plot_paths {
                let filename = std::path::Path::new(p).file_name().unwrap().to_str().unwrap();
                report.push_str(&format!("![{}]({})\n\n", filename, filename));
            }

            std::fs::write(&report_path, &report)?;
            println!("\nBenchmark report: {report_path}");
        }
    }

    Ok(())
}
