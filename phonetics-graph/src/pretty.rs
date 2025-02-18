/// Returns a prettified string of the given integer where every 3 digits
/// are separated by a comma.
///
/// From: https://stackoverflow.com/a/58437629/
pub fn prettified_int(i: u64) -> String {
    let mut s = String::new();
    let i_str = i.to_string();
    let a = i_str.chars().rev().enumerate();
    for (idx, val) in a {
        if idx != 0 && idx % 3 == 0 {
            s.insert(0, ',');
        }
        s.insert(0, val);
    }
    return s;
}
