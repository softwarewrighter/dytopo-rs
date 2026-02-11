use anyhow::Result;
use dytopo_core::Topology;
use std::fs::File;
use std::io::{BufWriter, Write};

pub fn write_dot(path: &str, topo: &Topology) -> Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    writeln!(w, "digraph round_{} {{", topo.round)?;
    writeln!(w, "  rankdir=LR;")?;
    for e in &topo.edges {
        writeln!(
            w,
            "  {} -> {} [label=\"{:.3}\"];",
            e.from.0, e.to.0, e.score
        )?;
    }
    writeln!(w, "}}")?;
    w.flush()?;
    Ok(())
}
