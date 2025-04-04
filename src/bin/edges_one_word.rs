use std::{
    cmp::Reverse,
    fmt,
    fs::File,
    io::{self, Read, Seek, Write},
};

const NUM_NODES: u32 = 100_000;

#[derive(Clone)]
struct EgoEdge {
    other_node: u32,
    weight: i8, // but they are stored as u8 in the file
}

struct Word {
    word: String,
    phonetics: String,
}

impl fmt::Display for Word {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} ({})", self.word, self.phonetics)
    }
}

static NODES_FILE: &str = "data/graph/nodes.csv";
static EDGES_FILE: &str = "data/graph/final/edges.gpu.bin";

fn read_node_labels() -> Vec<Word> {
    // csv file: id,label (phonetics)
    let file = File::open(NODES_FILE).expect("Failed to open nodes.csv");
    let mut words = Vec::new();
    let mut rdr = csv::Reader::from_reader(file);
    for result in rdr.records() {
        let record = result.expect("Failed to read record");
        let label = record.get(1).expect("Failed to get node label");
        if let Some((word, phonetics)) = label.rsplit_once(" (") {
            let phonetics = phonetics.trim_end_matches(')').to_string();
            words.push(Word {
                word: word.to_string(),
                phonetics,
            });
        }
    }
    words
}

fn to_index(row: u64, col: u64) -> u64 {
    ((col as f64) - 1.0 - 0.5 * row.pow(2) as f64 - row as f64 * (1.5 - NUM_NODES as f64)) as u64
}

fn main() {
    // Read node labels
    let words = read_node_labels();

    // Prompt user for a word
    print!("Enter a word: ");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let word = input.trim();

    // Find the node ID for the word
    let node_id = match words.iter().position(|w| w.word.eq_ignore_ascii_case(word)) {
        Some(id) => id as u32,
        None => {
            eprintln!("Word not found in the graph.");
            return;
        }
    };

    // Open the edges file
    let file = File::open(EDGES_FILE).expect("Failed to open edges file");
    let mut reader = io::BufReader::new(file);

    // Calculate the edges for the node's row and column
    let mut edges: Vec<EgoEdge> = Vec::new();

    fn read_edge_weight(reader: &mut io::BufReader<File>, index: u64) -> i8 {
        let mut buffer = [0; 1];
        reader
            .seek(io::SeekFrom::Start(index))
            .expect("Failed to seek in edges file");
        reader
            .read_exact(&mut buffer)
            .expect("Failed to read edge weight");
        buffer[0] as i8
    }

    // -> Traverse column
    for row in 0..node_id {
        let index = to_index(row as u64, node_id as u64);
        let weight = read_edge_weight(&mut reader, index);
        edges.push(EgoEdge {
            other_node: row,
            weight,
        });
    }

    // -> Traverse row
    for col in node_id + 1..NUM_NODES {
        let index = to_index(node_id as u64, col as u64);
        let weight = read_edge_weight(&mut reader, index);
        edges.push(EgoEdge {
            other_node: col,
            weight,
        });
    }

    // Sort edges by weight and select some
    edges.sort_by_key(|edge| Reverse(edge.weight));
    let filtered_edges: Vec<EgoEdge> = edges
        .iter()
        .filter(|edge| words[edge.other_node as usize].word.len() <= 13)
        .cloned()
        .collect();
    let best_edges = &filtered_edges[..filtered_edges.len().min(1000)];
    let user_word = &words[node_id as usize];

    println!("#Top edges for word \"{}\":", user_word);
    for edge in best_edges {
        println!(
            "-> {} (weight: {})",
            words[edge.other_node as usize], edge.weight
        );
    }
}
