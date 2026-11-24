//! Anchored regular expressions for family rules, with Python `re.fullmatch` semantics over a fixed subset.
//!
//! Supported: literals, `.` (any but `\n`), `\d` `\w` `\s` and escaped punctuation, classes `[...]` / `[^...]`
//! with ranges, groups `(...)` / `(?:...)` with `|`, greedy `* + ? {n} {n,} {n,m}`, a leading `^` and a trailing
//! `$`. Anything else is a syntax error, so a rules file cannot rely on semantics this engine lacks. Matching is a
//! Thompson-NFA simulation, linear in the name length.

use std::fmt;

const MAX_REPEAT: u32 = 1000;
const MAX_PROGRAM: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternError(pub String);

impl fmt::Display for PatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unsupported or invalid pattern: {}", self.0)
    }
}

impl std::error::Error for PatternError {}

#[derive(Debug, Clone)]
enum Ast {
    Empty,
    Char(char),
    Any,
    Class {
        negated: bool,
        ranges: Vec<(char, char)>,
    },
    Concat(Vec<Ast>),
    Alt(Vec<Ast>),
    Repeat {
        inner: Box<Ast>,
        min: u32,
        max: Option<u32>,
    },
}

#[derive(Debug, Clone)]
enum Inst {
    Char(char),
    Any,
    Class {
        negated: bool,
        ranges: Vec<(char, char)>,
    },
    Split(usize, usize),
    Jmp(usize),
    Match,
}

/// A compiled pattern.
#[derive(Debug, Clone)]
pub struct Pattern {
    source: String,
    prog: Vec<Inst>,
}

struct Parser<'a> {
    chars: Vec<char>,
    pos: usize,
    source: &'a str,
}

impl Parser<'_> {
    fn err<T>(&self, what: &str) -> Result<T, PatternError> {
        Err(PatternError(format!(
            "{what} at {} in {:?}",
            self.pos, self.source
        )))
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn alt(&mut self) -> Result<Ast, PatternError> {
        let mut branches = vec![self.concat()?];
        while self.peek() == Some('|') {
            self.pos += 1;
            branches.push(self.concat()?);
        }
        Ok(if branches.len() == 1 {
            branches.pop().expect("one branch")
        } else {
            Ast::Alt(branches)
        })
    }

    fn concat(&mut self) -> Result<Ast, PatternError> {
        let mut items = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            let atom = self.atom()?;
            items.push(self.quantified(atom)?);
        }
        Ok(match items.len() {
            0 => Ast::Empty,
            1 => items.pop().expect("one item"),
            _ => Ast::Concat(items),
        })
    }

    fn number(&mut self) -> Option<u32> {
        let start = self.pos;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.pos += 1;
        }
        if start == self.pos {
            return None;
        }
        self.chars[start..self.pos]
            .iter()
            .collect::<String>()
            .parse()
            .ok()
    }

    fn quantified(&mut self, atom: Ast) -> Result<Ast, PatternError> {
        let (min, max) = match self.peek() {
            Some('*') => (0, None),
            Some('+') => (1, None),
            Some('?') => (0, Some(1)),
            Some('{') => {
                self.pos += 1;
                let Some(min) = self.number() else {
                    return self.err("bad repetition");
                };
                let max = if self.peek() == Some(',') {
                    self.pos += 1;
                    self.number()
                } else {
                    Some(min)
                };
                if self.peek() != Some('}')
                    || max.is_some_and(|m| m < min)
                    || min.max(max.unwrap_or(0)) > MAX_REPEAT
                {
                    return self.err("bad repetition");
                }
                (min, max)
            }
            _ => return Ok(atom),
        };
        self.pos += 1;
        if matches!(self.peek(), Some('*' | '+' | '?' | '{')) {
            return self.err("nested or lazy quantifier");
        }
        Ok(Ast::Repeat {
            inner: Box::new(atom),
            min,
            max,
        })
    }

    fn escape(&mut self, in_class: bool) -> Result<Vec<(char, char)>, PatternError> {
        self.pos += 1;
        let Some(c) = self.peek() else {
            return self.err("trailing backslash");
        };
        self.pos += 1;
        Ok(match c {
            'd' => vec![('0', '9')],
            'w' => vec![('0', '9'), ('A', 'Z'), ('_', '_'), ('a', 'z')],
            's' => vec![(' ', ' '), ('\t', '\r')],
            c if c.is_ascii_punctuation() => vec![(c, c)],
            _ if in_class => return self.err("unsupported escape in class"),
            _ => return self.err("unsupported escape"),
        })
    }

    fn atom(&mut self) -> Result<Ast, PatternError> {
        let c = self.peek().expect("caller checked");
        match c {
            '(' => {
                self.pos += 1;
                if self.peek() == Some('?') {
                    if self.chars.get(self.pos + 1) != Some(&':') {
                        return self.err("unsupported group");
                    }
                    self.pos += 2;
                }
                let inner = self.alt()?;
                if self.peek() != Some(')') {
                    return self.err("unbalanced parenthesis");
                }
                self.pos += 1;
                Ok(inner)
            }
            '[' => self.class(),
            '.' => {
                self.pos += 1;
                Ok(Ast::Any)
            }
            '\\' => {
                let ranges = self.escape(false)?;
                Ok(match ranges.as_slice() {
                    [(a, b)] if a == b => Ast::Char(*a),
                    _ => Ast::Class {
                        negated: false,
                        ranges,
                    },
                })
            }
            '*' | '+' | '?' | '{' | '}' | ']' | '^' | '$' | ')' => {
                self.err("unexpected metacharacter")
            }
            c => {
                self.pos += 1;
                Ok(Ast::Char(c))
            }
        }
    }

    fn class(&mut self) -> Result<Ast, PatternError> {
        self.pos += 1;
        let negated = self.peek() == Some('^');
        if negated {
            self.pos += 1;
        }
        let mut ranges = Vec::new();
        let mut first = true;
        loop {
            let Some(c) = self.peek() else {
                return self.err("unterminated class");
            };
            if c == ']' && !first {
                self.pos += 1;
                break;
            }
            first = false;
            if c == '\\' {
                ranges.extend(self.escape(true)?);
                continue;
            }
            if c == '[' {
                return self.err("nested class");
            }
            self.pos += 1;
            let is_range =
                self.peek() == Some('-') && self.chars.get(self.pos + 1).is_some_and(|&n| n != ']');
            if is_range {
                let hi = self.chars[self.pos + 1];
                if hi == '\\' || hi == '[' || hi < c {
                    return self.err("bad class range");
                }
                self.pos += 2;
                ranges.push((c, hi));
            } else {
                ranges.push((c, c));
            }
        }
        Ok(Ast::Class { negated, ranges })
    }
}

fn emit(ast: &Ast, prog: &mut Vec<Inst>) -> Result<(), PatternError> {
    if prog.len() > MAX_PROGRAM {
        return Err(PatternError("pattern too large".into()));
    }
    match ast {
        Ast::Empty => {}
        Ast::Char(c) => prog.push(Inst::Char(*c)),
        Ast::Any => prog.push(Inst::Any),
        Ast::Class { negated, ranges } => prog.push(Inst::Class {
            negated: *negated,
            ranges: ranges.clone(),
        }),
        Ast::Concat(items) => {
            for item in items {
                emit(item, prog)?;
            }
        }
        Ast::Alt(branches) => {
            let mut jumps = Vec::new();
            for (i, branch) in branches.iter().enumerate() {
                if i + 1 < branches.len() {
                    let split = prog.len();
                    prog.push(Inst::Split(split + 1, 0));
                    emit(branch, prog)?;
                    jumps.push(prog.len());
                    prog.push(Inst::Jmp(0));
                    let next = prog.len();
                    prog[split] = Inst::Split(split + 1, next);
                } else {
                    emit(branch, prog)?;
                }
            }
            let end = prog.len();
            for j in jumps {
                prog[j] = Inst::Jmp(end);
            }
        }
        Ast::Repeat { inner, min, max } => {
            for _ in 0..*min {
                emit(inner, prog)?;
            }
            match max {
                None => {
                    let split = prog.len();
                    prog.push(Inst::Split(split + 1, 0));
                    emit(inner, prog)?;
                    prog.push(Inst::Jmp(split));
                    let end = prog.len();
                    prog[split] = Inst::Split(split + 1, end);
                }
                Some(max) => {
                    let mut splits = Vec::new();
                    for _ in *min..*max {
                        splits.push(prog.len());
                        prog.push(Inst::Split(0, 0));
                        emit(inner, prog)?;
                    }
                    let end = prog.len();
                    for s in splits {
                        prog[s] = Inst::Split(s + 1, end);
                    }
                }
            }
        }
    }
    Ok(())
}

impl Pattern {
    pub fn new(source: &str) -> Result<Pattern, PatternError> {
        let mut body = source.strip_prefix('^').unwrap_or(source);
        if body.ends_with('$') && !body.ends_with("\\$") {
            body = &body[..body.len() - 1];
        }
        let mut p = Parser {
            chars: body.chars().collect(),
            pos: 0,
            source,
        };
        let ast = p.alt()?;
        if p.pos != p.chars.len() {
            return p.err("unbalanced parenthesis");
        }
        let mut prog = Vec::new();
        emit(&ast, &mut prog)?;
        prog.push(Inst::Match);
        Ok(Pattern {
            source: source.to_string(),
            prog,
        })
    }

    pub fn as_str(&self) -> &str {
        &self.source
    }

    fn add(&self, list: &mut Vec<usize>, seen: &mut [u32], stamp: u32, pc: usize) {
        if seen[pc] == stamp {
            return;
        }
        seen[pc] = stamp;
        match self.prog[pc] {
            Inst::Split(a, b) => {
                self.add(list, seen, stamp, a);
                self.add(list, seen, stamp, b);
            }
            Inst::Jmp(t) => self.add(list, seen, stamp, t),
            _ => list.push(pc),
        }
    }

    /// Python `re.fullmatch(pattern, text) is not None`. The state lists and the seen set are this thread's
    /// [`Scratch`], reused across calls (VALIDATE matches every tensor name against every rule, m216 design note L5).
    pub fn full_match(&self, text: &str) -> bool {
        SCRATCH.with(|s| {
            let Scratch {
                current,
                next,
                seen,
                stamp,
            } = &mut *s.borrow_mut();
            if seen.len() < self.prog.len() {
                seen.resize(self.prog.len(), 0);
            }
            current.clear();
            *stamp = bump(seen, *stamp);
            self.add(current, seen, *stamp, 0);
            for c in text.chars() {
                next.clear();
                *stamp = bump(seen, *stamp);
                for &pc in current.iter() {
                    let ok = match &self.prog[pc] {
                        Inst::Char(x) => *x == c,
                        Inst::Any => c != '\n',
                        Inst::Class { negated, ranges } => {
                            ranges.iter().any(|&(a, b)| a <= c && c <= b) != *negated
                        }
                        Inst::Match => false,
                        Inst::Split(..) | Inst::Jmp(_) => {
                            unreachable!("closure adds only consuming states")
                        }
                    };
                    if ok {
                        self.add(next, seen, *stamp, pc + 1);
                    }
                }
                if next.is_empty() {
                    return false;
                }
                std::mem::swap(current, next);
            }
            current
                .iter()
                .any(|&pc| matches!(self.prog[pc], Inst::Match))
        })
    }
}

/// One thread's matcher state: the current and next state lists and the seen set, stamped per step (a state is seen
/// in this step when `seen[pc] == stamp`), so no step clears or allocates.
#[derive(Default)]
struct Scratch {
    current: Vec<usize>,
    next: Vec<usize>,
    seen: Vec<u32>,
    stamp: u32,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::default();
}

/// The next step's stamp; on wrap-around every entry is cleared first, so an old stamp never reads as seen.
fn bump(seen: &mut [u32], stamp: u32) -> u32 {
    if stamp == u32::MAX {
        seen.fill(0);
        1
    } else {
        stamp + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected values are Python `bool(re.fullmatch(pattern, name))`.
    #[test]
    fn agrees_with_python_fullmatch() {
        let quad = r"model\.layers\.\d+\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight";
        let cases: &[(&str, &str, bool)] = &[
            (quad, "model.layers.12.mlp.experts.7.up_proj.weight", true),
            (
                quad,
                "model.layers.12.mlp.experts.7.up_proj.weight_scale",
                false,
            ),
            (r"^mtp\..*$", "mtp.layers.0.x", true),
            (r"mtp\..*", "xmtp.a", false),
            (r"a[^.]+b", "axyb", true),
            (r"a[^.]+b", "ax.b", false),
            (r"(?:ab|a)c?d{2,3}", "abdd", true),
            (r"(?:ab|a)c?d{2,3}", "acdddd", false),
            (r"x(a|ab)*c", "xababac", true),
            (r"[A-Za-z_]\w*", "_a9", true),
            (r"[a-c-]+", "a-c", true),
            (r"a{2}", "aa", true),
            (r"a{2,}", "aaaaa", true),
            (r"(a*)*b", "aaaaaaaaaaaaaaaaaaaaaaaac", false),
            (r"\[x\]", "[x]", true),
            (r"", "", true),
            (r"a|", "", true),
        ];
        for (p, s, want) in cases {
            let re = Pattern::new(p).unwrap_or_else(|e| panic!("{p}: {e}"));
            assert_eq!(re.full_match(s), *want, "{p} on {s}");
        }
    }

    /// The reused scratch (L5) gives the fresh-state answer for interleaved patterns of different sizes and across
    /// the stamp's wrap-around.
    #[test]
    fn reused_scratch_agrees_across_patterns_and_stamp_wraparound() {
        let quad =
            Pattern::new(r"model\.layers\.\d+\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight")
                .unwrap();
        let tiny = Pattern::new(r"x(a|ab)*c").unwrap();
        let name = "model.layers.12.mlp.experts.7.up_proj.weight";
        for start in [0, u32::MAX - 3, u32::MAX - 60, u32::MAX] {
            SCRATCH.with(|s| {
                let mut s = s.borrow_mut();
                s.seen.fill(0);
                s.stamp = start;
            });
            for _ in 0..3 {
                assert!(quad.full_match(name), "start {start}");
                assert!(!quad.full_match("model.layers.12.mlp.experts.7.up_proj.weight_scale"));
                assert!(tiny.full_match("xababac"), "start {start}");
                assert!(!tiny.full_match("xabbc"), "start {start}");
                assert!(!quad.full_match("model.layers.x"), "start {start}");
            }
        }
    }

    #[test]
    fn rejects_syntax_outside_the_subset() {
        for p in [
            "(a",
            "a)",
            "[a",
            "a**",
            "*a",
            r"(?=a)",
            r"(?P<n>a)",
            r"\1",
            "a{3,2}",
            "a{x}",
            r"a\",
            "a+?",
            "a^",
            "$a",
            r"\bx",
            "a{100000}",
        ] {
            assert!(Pattern::new(p).is_err(), "{p} should be rejected");
        }
    }
}
