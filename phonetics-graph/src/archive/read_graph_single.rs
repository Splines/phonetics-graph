use std::{fs::File, io::Read};

const MAX_NODE_ID: u32 = 245645;
const NODE_ID: u32 = 159468;

struct Edge {
    source: u32,
    target: u32,
    weight: i8,
}

/**
 * Read the edge binary file and convert to a list of edges.
 */
fn read_edges(buffer: &Vec<u8>) -> Vec<Edge> {
    // The graph.edges file is a binary file that holds weights for edges of a graph.
    // The file itself just contains a list of bytes, where each byte represents the weight of an edge.
    // The node ids are implicit in the order of the edges:
    // 0,1,score -> 0,2,score -> 0,3,score -> 1,1,score -> 1,2,score -> 1,3,score
    // here 3 is the maximum node id, which should be a parameter of the program.
    // Also take into account that the graph is fully-connected and the edges
    // are undirected, i.e. we only calculated scores for the cases i <= j.
    let mut edges = Vec::new();
    let mut index = 0;

    let i = NODE_ID;

    for j in 0..=MAX_NODE_ID {
        if index >= buffer.len() {
            break;
        }

        let weight = buffer[index] as i8;
        edges.push(Edge {
            source: i as u32,
            target: j as u32,
            weight,
        });
        // also add the reverse edge (edges are undirected)
        if i != j {
            edges.push(Edge {
                source: j as u32,
                target: i as u32,
                weight,
            });
        }
        index += 1;
    }

    edges
}

fn read_node_labels() -> Vec<String> {
    // csv file has data: id,label
    let file = File::open("../data/ipa/graph-rust/nodes.csv").expect("Failed to open nodes.csv");
    let mut labels = Vec::new();
    let mut rdr = csv::Reader::from_reader(file);
    for result in rdr.records() {
        let record = result.expect("Failed to read record");
        let label = record.get(1).expect("Failed to get node label");
        labels.push(label.to_string());
    }
    labels
}

// Read the edge binary file and convert to a list of edges
fn main() {
    let mut file =
        File::open("../data/ipa/graph-rust/edges.bin").expect("Failed to open edges.bin");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .expect("Failed to read edges.bin");
    let edges = read_edges(&buffer);
    let node_labels = read_node_labels();
    println!("✅ Done reading edges.bin and nodes.csv");

    // For the node with id 8, find the nodes with the highest edge weights
    // Note that this a fully-connected graph.
    let mut top_edges = edges
        .iter()
        .filter(|edge| edge.source == NODE_ID)
        .collect::<Vec<&Edge>>();
    top_edges.sort_by(|a, b| b.weight.cmp(&a.weight));

    println!("😇 Top edges for node {}: ", NODE_ID);
    for edge in top_edges.iter().take(150) {
        let target_label = &node_labels[edge.target as usize];
        println!(
            "{},{},{}: {}",
            edge.source, edge.target, edge.weight, target_label
        );
    }

    // Print worst edges/nodes
    println!("😅 Worst edges for node {}: ", NODE_ID);
    for edge in top_edges.iter().rev().take(30) {
        let target_label = &node_labels[edge.target as usize];
        println!(
            "{},{},{}: {}",
            edge.source, edge.target, edge.weight, target_label
        );
    }
}
