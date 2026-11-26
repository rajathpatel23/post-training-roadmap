/// One node in the tape.
/// `data` is the forward value. `grad` is d(loss) / d(data), not a derivative
/// of `data` itself. It stays 0 until backward accumulates into it.
struct Value {
    data: f64,
    grad: f64,
}

/// Owns every node. Callers hold a `usize` index, not the `Value`.
/// The index is `Copy`, so it can be passed into `add` and used again
/// afterward. The `Value` stays in this list.
struct Tape {
    nodes: Vec<Value>,
}

impl Tape {
    fn new() -> Tape {
        Tape { nodes: Vec::new() }
    }

    /// Append a leaf. The returned id is the index this node occupies,
    /// which is the length of the list before the push.
    /// `&mut self` is required because the list grows.
    fn var(&mut self, data: f64) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Value { data, grad: 0.0 });
        id
    }

    /// Read two nodes, push their sum, and return the new index.
    /// `x` and `y` are not stored on the new node. Backward still needs
    /// those parent indexes, and it needs to know this op was an add,
    /// so it can send the upstream grad to both parents unchanged.
    fn add(&mut self, x: usize, y: usize) -> usize {
        let data = self.nodes[x].data + self.nodes[y].data;
        self.var(data)
    }
}


fn main() {
    let mut tape = Tape::new();
    // `a`, `b`, and `c` are indexes into `tape.nodes`.
    let a = tape.var(1.0);
    let b = tape.var(2.0);
    let c = tape.add(a, b);
    println!(
        "{} {} {}",
        tape.nodes[a].data,
        tape.nodes[b].data,
        tape.nodes[c].data
    );
}
