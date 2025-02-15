use std::{fs::File, io::Read};

const NUM_NODES: u32 = 100000;

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

    for i in 0..NUM_NODES {
        for j in i..NUM_NODES {
            if index >= buffer.len() {
                break;
            }

            let weight = buffer[index] as i8;

            let is_interesting_edge = weight > 4 && weight < 10;
            if !is_interesting_edge {
                index += 1;
                continue;
            }

            edges.push(Edge {
                source: i as u32,
                target: j as u32,
                weight,
            });
            // also add the reverse edge (edges are undirected)
            // if i != j {
            //     edges.push(Edge {
            //         source: j as u32,
            //         target: i as u32,
            //         weight,
            //     });
            // }
            index += 1;
        }
    }

    edges
}

#[allow(dead_code)]
fn read_node_labels() -> Vec<String> {
    // csv file has data: id,label
    let file = File::open("../data/graph/nodes.csv").expect("Failed to open nodes.csv");
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
    let mut file = File::open("../data/graph/edges-new-gpu.bin").expect("Failed to open edges.bin");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .expect("Failed to read edges.bin");

    let first_few_bytes: Vec<i8> = buffer[..20].iter().map(|&x| x as i8).collect();
    println!("First few bytes of the buffer: {:?}", first_few_bytes);
    let min_weight = buffer.iter().map(|&x| x as i8).min().unwrap();
    let max_weight = buffer.iter().map(|&x| x as i8).max().unwrap();
    println!("Minimum weight: {}", min_weight);
    println!("Maximum weight: {}", max_weight);

    let edges = read_edges(&buffer);
    println!("✅ Done reading edges.bin and nodes.csv");

    // Store the edges in a csv file source,target,weight
    let mut wtr = csv::Writer::from_path("../data/graph/edges-new-gpu.csv")
        .expect("Failed to create edges.csv");
    wtr.write_record(&["source", "target", "weight"])
        .expect("Failed to write header");
    for edge in edges {
        wtr.write_record(&[
            edge.source.to_string(),
            edge.target.to_string(),
            edge.weight.to_string(),
        ])
        .expect("Failed to write record");
    }
}
