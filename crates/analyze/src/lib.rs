//! Trace analysis and metrics for DyTopo runs.

use anyhow::Result;
use dytopo_core::{Edge, TraceEvent};
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Parsed trace data organized by round.
#[derive(Debug, Default)]
pub struct ParsedTrace {
    pub rounds: Vec<RoundData>,
    pub agent_count: usize,
}

#[derive(Debug, Default, Clone)]
pub struct RoundData {
    pub round: usize,
    pub goal: String,
    pub start_ts: u64,
    pub end_ts: u64,
    pub agents: Vec<AgentData>,
    pub edges: Vec<Edge>,
}

#[derive(Debug, Default, Clone)]
pub struct AgentData {
    pub agent_id: usize,
    pub query: String,
    pub key: String,
    pub draft: String,
    pub tokens_in: usize,
    pub tokens_out: usize,
}

/// Load and parse a JSONL trace file.
pub fn load_trace(path: &str) -> Result<ParsedTrace> {
    let f = File::open(path)?;
    let reader = BufReader::new(f);

    let mut trace = ParsedTrace::default();
    let mut current_round: Option<RoundData> = None;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let event: TraceEvent = serde_json::from_str(&line)?;

        match event {
            TraceEvent::RoundStart {
                round,
                goal,
                agent_count,
                ts_unix_ms,
            } => {
                if let Some(rd) = current_round.take() {
                    trace.rounds.push(rd);
                }
                trace.agent_count = agent_count;
                current_round = Some(RoundData {
                    round,
                    goal,
                    start_ts: ts_unix_ms,
                    ..Default::default()
                });
            }
            TraceEvent::AgentIO {
                agent_id,
                query,
                key,
                draft,
                tokens_in,
                tokens_out,
                ..
            } => {
                if let Some(ref mut rd) = current_round {
                    rd.agents.push(AgentData {
                        agent_id,
                        query,
                        key,
                        draft,
                        tokens_in,
                        tokens_out,
                    });
                }
            }
            TraceEvent::Topology { edges, .. } => {
                if let Some(ref mut rd) = current_round {
                    rd.edges = edges;
                }
            }
            TraceEvent::RoundEnd { ts_unix_ms, .. } => {
                if let Some(ref mut rd) = current_round {
                    rd.end_ts = ts_unix_ms;
                }
            }
            TraceEvent::Message { .. } => {
                // Messages are derivable from edges + agents
            }
            TraceEvent::QualityEval { .. } => {
                // Quality evals are stored separately
            }
        }
    }

    if let Some(rd) = current_round {
        trace.rounds.push(rd);
    }

    Ok(trace)
}

/// Metrics computed from a trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceMetrics {
    pub trace_path: String,
    pub agent_count: usize,
    pub round_count: usize,
    pub rounds: Vec<RoundMetrics>,
    pub summary: SummaryMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundMetrics {
    pub round: usize,
    pub duration_ms: u64,
    pub edge_count: usize,
    pub avg_score: f32,
    pub min_score: f32,
    pub max_score: f32,
    pub score_std: f32,
    pub graph_density: f32,
    pub avg_in_degree: f32,
    pub avg_out_degree: f32,
    pub query_avg_len: f32,
    pub key_avg_len: f32,
    pub draft_avg_len: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryMetrics {
    pub total_duration_ms: u64,
    pub avg_round_duration_ms: f32,
    pub total_edges: usize,
    pub avg_edges_per_round: f32,
    pub overall_avg_score: f32,
    pub score_trend: f32, // positive = scores improving over rounds
    pub edge_stability: f32, // fraction of edges that persist between rounds
}

/// Compute metrics from a parsed trace.
pub fn compute_metrics(trace: &ParsedTrace, trace_path: &str) -> TraceMetrics {
    let mut rounds = Vec::new();
    let n = trace.agent_count;
    let max_edges = if n > 1 { n * (n - 1) } else { 1 };

    for rd in &trace.rounds {
        let duration_ms = rd.end_ts.saturating_sub(rd.start_ts);
        let edge_count = rd.edges.len();

        let scores: Vec<f32> = rd.edges.iter().map(|e| e.score).collect();
        let avg_score = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        };
        let min_score = scores.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let variance = if scores.len() > 1 {
            scores.iter().map(|s| (s - avg_score).powi(2)).sum::<f32>() / scores.len() as f32
        } else {
            0.0
        };
        let score_std = variance.sqrt();

        let graph_density = edge_count as f32 / max_edges as f32;

        // In-degree and out-degree
        let mut in_deg: HashMap<usize, usize> = HashMap::new();
        let mut out_deg: HashMap<usize, usize> = HashMap::new();
        for e in &rd.edges {
            *out_deg.entry(e.from.0).or_insert(0) += 1;
            *in_deg.entry(e.to.0).or_insert(0) += 1;
        }
        let avg_in_degree = if n > 0 {
            in_deg.values().sum::<usize>() as f32 / n as f32
        } else {
            0.0
        };
        let avg_out_degree = if n > 0 {
            out_deg.values().sum::<usize>() as f32 / n as f32
        } else {
            0.0
        };

        // Text lengths
        let query_lens: Vec<usize> = rd.agents.iter().map(|a| a.query.len()).collect();
        let key_lens: Vec<usize> = rd.agents.iter().map(|a| a.key.len()).collect();
        let draft_lens: Vec<usize> = rd.agents.iter().map(|a| a.draft.len()).collect();

        let query_avg_len = if query_lens.is_empty() {
            0.0
        } else {
            query_lens.iter().sum::<usize>() as f32 / query_lens.len() as f32
        };
        let key_avg_len = if key_lens.is_empty() {
            0.0
        } else {
            key_lens.iter().sum::<usize>() as f32 / key_lens.len() as f32
        };
        let draft_avg_len = if draft_lens.is_empty() {
            0.0
        } else {
            draft_lens.iter().sum::<usize>() as f32 / draft_lens.len() as f32
        };

        rounds.push(RoundMetrics {
            round: rd.round,
            duration_ms,
            edge_count,
            avg_score,
            min_score: if min_score.is_finite() { min_score } else { 0.0 },
            max_score: if max_score.is_finite() { max_score } else { 0.0 },
            score_std,
            graph_density,
            avg_in_degree,
            avg_out_degree,
            query_avg_len,
            key_avg_len,
            draft_avg_len,
        });
    }

    // Summary metrics
    let total_duration_ms: u64 = rounds.iter().map(|r| r.duration_ms).sum();
    let avg_round_duration_ms = if rounds.is_empty() {
        0.0
    } else {
        total_duration_ms as f32 / rounds.len() as f32
    };

    let total_edges: usize = rounds.iter().map(|r| r.edge_count).sum();
    let avg_edges_per_round = if rounds.is_empty() {
        0.0
    } else {
        total_edges as f32 / rounds.len() as f32
    };

    let all_scores: Vec<f32> = rounds.iter().map(|r| r.avg_score).collect();
    let overall_avg_score = if all_scores.is_empty() {
        0.0
    } else {
        all_scores.iter().sum::<f32>() / all_scores.len() as f32
    };

    // Score trend: linear regression slope
    let score_trend = if rounds.len() >= 2 {
        let n = rounds.len() as f32;
        let sum_x: f32 = (0..rounds.len()).map(|i| i as f32).sum();
        let sum_y: f32 = all_scores.iter().sum();
        let sum_xy: f32 = rounds
            .iter()
            .enumerate()
            .map(|(i, r)| i as f32 * r.avg_score)
            .sum();
        let sum_x2: f32 = (0..rounds.len()).map(|i| (i as f32).powi(2)).sum();
        (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2))
    } else {
        0.0
    };

    // Edge stability: Jaccard similarity between consecutive rounds
    let edge_stability = if trace.rounds.len() >= 2 {
        let mut jaccard_sum = 0.0;
        for i in 1..trace.rounds.len() {
            let prev: std::collections::HashSet<_> = trace.rounds[i - 1]
                .edges
                .iter()
                .map(|e| (e.from.0, e.to.0))
                .collect();
            let curr: std::collections::HashSet<_> = trace.rounds[i]
                .edges
                .iter()
                .map(|e| (e.from.0, e.to.0))
                .collect();
            let intersection = prev.intersection(&curr).count();
            let union = prev.union(&curr).count();
            if union > 0 {
                jaccard_sum += intersection as f32 / union as f32;
            }
        }
        jaccard_sum / (trace.rounds.len() - 1) as f32
    } else {
        1.0
    };

    TraceMetrics {
        trace_path: trace_path.to_string(),
        agent_count: trace.agent_count,
        round_count: trace.rounds.len(),
        rounds,
        summary: SummaryMetrics {
            total_duration_ms,
            avg_round_duration_ms,
            total_edges,
            avg_edges_per_round,
            overall_avg_score,
            score_trend,
            edge_stability,
        },
    }
}

/// Generate a text report from metrics.
pub fn format_report(metrics: &TraceMetrics) -> String {
    let mut s = String::new();

    s.push_str(&format!("# Trace Analysis: {}\n\n", metrics.trace_path));
    s.push_str(&format!(
        "**Agents:** {} | **Rounds:** {}\n\n",
        metrics.agent_count, metrics.round_count
    ));

    s.push_str("## Summary\n\n");
    s.push_str("| Metric | Value |\n|--------|-------|\n");
    s.push_str(&format!(
        "| Total Duration | {:.0}ms |\n",
        metrics.summary.total_duration_ms
    ));
    s.push_str(&format!(
        "| Avg Round Duration | {:.0}ms |\n",
        metrics.summary.avg_round_duration_ms
    ));
    s.push_str(&format!(
        "| Total Edges | {} |\n",
        metrics.summary.total_edges
    ));
    s.push_str(&format!(
        "| Avg Edges/Round | {:.1} |\n",
        metrics.summary.avg_edges_per_round
    ));
    s.push_str(&format!(
        "| Overall Avg Score | {:.3} |\n",
        metrics.summary.overall_avg_score
    ));
    s.push_str(&format!(
        "| Score Trend | {:+.4}/round |\n",
        metrics.summary.score_trend
    ));
    s.push_str(&format!(
        "| Edge Stability | {:.1}% |\n",
        metrics.summary.edge_stability * 100.0
    ));

    s.push_str("\n## Per-Round Metrics\n\n");
    s.push_str("| Round | Duration | Edges | Avg Score | Min | Max | Density |\n");
    s.push_str("|-------|----------|-------|-----------|-----|-----|---------|\n");
    for r in &metrics.rounds {
        s.push_str(&format!(
            "| {} | {}ms | {} | {:.3} | {:.3} | {:.3} | {:.1}% |\n",
            r.round,
            r.duration_ms,
            r.edge_count,
            r.avg_score,
            r.min_score,
            r.max_score,
            r.graph_density * 100.0
        ));
    }

    s
}

/// Plot edge score distribution across rounds as SVG.
pub fn plot_scores(metrics: &TraceMetrics, output_path: &str) -> Result<()> {
    let root = SVGBackend::new(output_path, (800, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_round = metrics.round_count.saturating_sub(1) as f32;
    let max_score = 1.0f32;

    let mut chart = ChartBuilder::on(&root)
        .caption("Edge Score Evolution", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f32..max_round.max(1.0), 0f32..max_score)?;

    chart
        .configure_mesh()
        .x_desc("Round")
        .y_desc("Score")
        .draw()?;

    // Avg score line
    let avg_data: Vec<(f32, f32)> = metrics
        .rounds
        .iter()
        .map(|r| (r.round as f32, r.avg_score))
        .collect();

    chart
        .draw_series(LineSeries::new(avg_data.clone(), &BLUE))?
        .label("Avg Score")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    // Min/max as area
    let min_max: Vec<(f32, f32, f32)> = metrics
        .rounds
        .iter()
        .map(|r| (r.round as f32, r.min_score, r.max_score))
        .collect();

    chart.draw_series(min_max.iter().map(|(x, min, max)| {
        Rectangle::new([(*x - 0.1, *min), (*x + 0.1, *max)], BLUE.mix(0.3).filled())
    }))?;

    // Points
    chart.draw_series(PointSeries::of_element(
        avg_data,
        5,
        &BLUE,
        &|c, s, st| Circle::new(c, s, st.filled()),
    ))?;

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

/// Plot round timing comparison as bar chart.
pub fn plot_timing(metrics: &TraceMetrics, output_path: &str) -> Result<()> {
    let root = SVGBackend::new(output_path, (800, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_time = metrics
        .rounds
        .iter()
        .map(|r| r.duration_ms)
        .max()
        .unwrap_or(1000) as f32;

    let mut chart = ChartBuilder::on(&root)
        .caption("Round Duration", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0..(metrics.round_count as i32),
            0f32..max_time * 1.1,
        )?;

    chart
        .configure_mesh()
        .x_desc("Round")
        .y_desc("Duration (ms)")
        .draw()?;

    chart.draw_series(
        metrics
            .rounds
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let x0 = i as i32;
                let x1 = i as i32 + 1;
                Rectangle::new(
                    [(x0, 0f32), (x1, r.duration_ms as f32)],
                    GREEN.mix(0.8).filled(),
                )
            }),
    )?;

    root.present()?;
    Ok(())
}

/// Plot graph density evolution.
pub fn plot_density(metrics: &TraceMetrics, output_path: &str) -> Result<()> {
    let root = SVGBackend::new(output_path, (800, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_round = metrics.round_count.saturating_sub(1) as f32;

    let mut chart = ChartBuilder::on(&root)
        .caption("Graph Density Over Rounds", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f32..max_round.max(1.0), 0f32..1f32)?;

    chart
        .configure_mesh()
        .x_desc("Round")
        .y_desc("Density")
        .draw()?;

    let data: Vec<(f32, f32)> = metrics
        .rounds
        .iter()
        .map(|r| (r.round as f32, r.graph_density))
        .collect();

    chart
        .draw_series(LineSeries::new(data.clone(), &RED))?
        .label("Density")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart.draw_series(PointSeries::of_element(
        data,
        5,
        &RED,
        &|c, s, st| Circle::new(c, s, st.filled()),
    ))?;

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

/// Generate all plots for a trace.
pub fn generate_all_plots(metrics: &TraceMetrics, output_dir: &str) -> Result<Vec<String>> {
    std::fs::create_dir_all(output_dir)?;

    let mut paths = Vec::new();

    let scores_path = format!("{}/scores.svg", output_dir);
    plot_scores(metrics, &scores_path)?;
    paths.push(scores_path);

    let timing_path = format!("{}/timing.svg", output_dir);
    plot_timing(metrics, &timing_path)?;
    paths.push(timing_path);

    let density_path = format!("{}/density.svg", output_dir);
    plot_density(metrics, &density_path)?;
    paths.push(density_path);

    Ok(paths)
}

// ============================================================================
// Quality Evaluation
// ============================================================================

/// Quality evaluation result from LLM-as-judge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityEval {
    pub task: String,
    pub solution: String,
    pub score: f32,           // 1-10 scale
    pub correctness: f32,     // 1-10 subscale
    pub completeness: f32,    // 1-10 subscale
    pub clarity: f32,         // 1-10 subscale
    pub reasoning: String,
    pub tokens_total: usize,
}

/// Extract the combined solution from the final round's agent drafts.
pub fn extract_solution(trace: &ParsedTrace) -> Option<String> {
    let last_round = trace.rounds.last()?;

    // Combine all agent drafts from the final round
    let mut solution = String::new();
    for agent in &last_round.agents {
        if !solution.is_empty() {
            solution.push_str("\n\n---\n\n");
        }
        solution.push_str(&format!("Agent {} contribution:\n{}", agent.agent_id, agent.draft));
    }

    Some(solution)
}

/// Compute total tokens used across all rounds.
pub fn compute_total_tokens(trace: &ParsedTrace) -> usize {
    trace.rounds.iter()
        .flat_map(|r| r.agents.iter())
        .map(|a| a.tokens_in + a.tokens_out)
        .sum()
}

/// Use LLM-as-judge to evaluate solution quality.
/// Returns a QualityEval with scores on a 1-10 scale.
pub fn evaluate_quality(
    task: &str,
    solution: &str,
    tokens_total: usize,
    llm: &dyn dytopo_llm::LlmClient,
) -> Result<QualityEval> {
    let prompt = format!(
        r#"You are an expert evaluator. Rate the following solution to a task.

TASK: {task}

SOLUTION:
{solution}

Rate the solution on these criteria (1-10 scale, 10 = excellent):
1. CORRECTNESS: Does it solve the problem correctly?
2. COMPLETENESS: Does it cover all aspects of the task?
3. CLARITY: Is it well-explained and easy to understand?

Respond in this exact JSON format:
{{
  "correctness": <1-10>,
  "completeness": <1-10>,
  "clarity": <1-10>,
  "reasoning": "<brief explanation of your ratings>"
}}

Be strict but fair. A score of 5 is average, 7+ is good, 9+ is excellent."#
    );

    let response = llm.complete(&prompt)?;

    // Parse the JSON response
    let json: serde_json::Value = dytopo_llm::extract_json(&response)?;

    let correctness = json["correctness"].as_f64().unwrap_or(5.0) as f32;
    let completeness = json["completeness"].as_f64().unwrap_or(5.0) as f32;
    let clarity = json["clarity"].as_f64().unwrap_or(5.0) as f32;
    let reasoning = json["reasoning"].as_str().unwrap_or("No reasoning provided").to_string();

    // Overall score is average of subscales
    let score = (correctness + completeness + clarity) / 3.0;

    Ok(QualityEval {
        task: task.to_string(),
        solution: solution.to_string(),
        score,
        correctness,
        completeness,
        clarity,
        reasoning,
        tokens_total,
    })
}

/// Comparison result between two runs (e.g., dynamic vs baseline).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub name_a: String,
    pub name_b: String,
    pub task: String,
    pub quality_a: QualityEval,
    pub quality_b: QualityEval,
    pub metrics_a: TraceMetrics,
    pub metrics_b: TraceMetrics,
}

impl ComparisonResult {
    /// Quality difference (A - B). Positive means A is better.
    pub fn quality_diff(&self) -> f32 {
        self.quality_a.score - self.quality_b.score
    }

    /// Token efficiency: quality per 1000 tokens.
    pub fn efficiency_a(&self) -> f32 {
        if self.quality_a.tokens_total == 0 {
            0.0
        } else {
            self.quality_a.score / (self.quality_a.tokens_total as f32 / 1000.0)
        }
    }

    pub fn efficiency_b(&self) -> f32 {
        if self.quality_b.tokens_total == 0 {
            0.0
        } else {
            self.quality_b.score / (self.quality_b.tokens_total as f32 / 1000.0)
        }
    }

    /// Generate markdown comparison report.
    pub fn to_markdown(&self) -> String {
        let mut s = String::new();

        s.push_str(&format!("# Comparison: {} vs {}\n\n", self.name_a, self.name_b));
        s.push_str(&format!("**Task:** {}\n\n", self.task));

        s.push_str("## Quality Scores (1-10 scale)\n\n");
        s.push_str("| Metric | {} | {} | Winner |\n");
        s.push_str("|--------|");
        s.push_str(&format!("{}|", self.name_a));
        s.push_str(&format!("{}|", self.name_b));
        s.push_str("--------|\n");

        let winner = |a: f32, b: f32| -> &str {
            if a > b + 0.5 { &self.name_a }
            else if b > a + 0.5 { &self.name_b }
            else { "Tie" }
        };

        s.push_str(&format!(
            "| Overall | {:.1} | {:.1} | {} |\n",
            self.quality_a.score, self.quality_b.score,
            winner(self.quality_a.score, self.quality_b.score)
        ));
        s.push_str(&format!(
            "| Correctness | {:.1} | {:.1} | {} |\n",
            self.quality_a.correctness, self.quality_b.correctness,
            winner(self.quality_a.correctness, self.quality_b.correctness)
        ));
        s.push_str(&format!(
            "| Completeness | {:.1} | {:.1} | {} |\n",
            self.quality_a.completeness, self.quality_b.completeness,
            winner(self.quality_a.completeness, self.quality_b.completeness)
        ));
        s.push_str(&format!(
            "| Clarity | {:.1} | {:.1} | {} |\n",
            self.quality_a.clarity, self.quality_b.clarity,
            winner(self.quality_a.clarity, self.quality_b.clarity)
        ));

        s.push_str("\n## Efficiency\n\n");
        s.push_str("| Metric | {} | {} |\n");
        s.push_str(&format!("|--------|{}|{}|\n", self.name_a, self.name_b));
        s.push_str(&format!(
            "| Total Tokens | {} | {} |\n",
            self.quality_a.tokens_total, self.quality_b.tokens_total
        ));
        s.push_str(&format!(
            "| Total Edges | {} | {} |\n",
            self.metrics_a.summary.total_edges, self.metrics_b.summary.total_edges
        ));
        s.push_str(&format!(
            "| Quality/1K Tokens | {:.2} | {:.2} |\n",
            self.efficiency_a(), self.efficiency_b()
        ));

        s.push_str("\n## Reasoning\n\n");
        s.push_str(&format!("**{}:** {}\n\n", self.name_a, self.quality_a.reasoning));
        s.push_str(&format!("**{}:** {}\n\n", self.name_b, self.quality_b.reasoning));

        s
    }
}

/// Plot quality comparison as bar chart.
pub fn plot_quality_comparison(results: &[(&str, f32)], output_path: &str) -> Result<()> {
    let root = SVGBackend::new(output_path, (800, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_score = 10.0f32;
    let n = results.len();

    let mut chart = ChartBuilder::on(&root)
        .caption("Quality Scores by Topology", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(50)
        .build_cartesian_2d(0..(n as i32), 0f32..max_score)?;

    chart
        .configure_mesh()
        .x_desc("Topology")
        .y_desc("Quality Score (1-10)")
        .x_labels(n)
        .x_label_formatter(&|i| {
            results.get(*i as usize).map(|(name, _)| name.to_string()).unwrap_or_default()
        })
        .draw()?;

    // Draw bars with different colors
    let colors = [BLUE, GREEN, RED, CYAN, MAGENTA];
    for (i, (_, score)) in results.iter().enumerate() {
        let color = colors[i % colors.len()];
        chart.draw_series(std::iter::once(Rectangle::new(
            [(i as i32, 0f32), (i as i32 + 1, *score)],
            color.mix(0.8).filled(),
        )))?;
    }

    root.present()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_trace_empty() {
        // Can't test without a file, but verify struct defaults
        let trace = ParsedTrace::default();
        assert_eq!(trace.rounds.len(), 0);
    }

    #[test]
    fn test_extract_solution_empty() {
        let trace = ParsedTrace::default();
        assert!(extract_solution(&trace).is_none());
    }
}
