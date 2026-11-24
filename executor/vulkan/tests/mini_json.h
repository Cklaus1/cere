// A minimal JSON reader for the Vulkan host tests (safetensors headers, manifest.json, the inventory JSON): objects,
// arrays, strings (\uXXXX kept as '?'), integers (exact as int64/uint64), other numbers as double, true/false/null.
// Not a validator (AGENTS.md 11: json parsing is not schema validation): the tests check the fields they read.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cere::testjson {

struct Value {
    enum Kind { kNull, kBool, kInt, kNum, kStr, kArr, kObj } kind = kNull;
    bool b = false;
    uint64_t u = 0;  // kInt: the value when non-negative
    int64_t i = 0;
    double d = 0;
    std::string s;
    std::vector<Value> a;
    std::vector<std::pair<std::string, Value>> o;  // file order kept

    const Value *get(const std::string &key) const {
        for (const auto &kv : o) {
            if (kv.first == key) return &kv.second;
        }
        return nullptr;
    }
};

class Parser {
public:
    Parser(const char *p, size_t n) : p_(p), end_(p + n) {}
    bool parse(Value *out) {
        ws();
        if (!value(out)) return false;
        ws();
        return p_ == end_;
    }

private:
    const char *p_;
    const char *end_;

    void ws() {
        while (p_ < end_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) ++p_;
    }
    bool lit(const char *w) {
        const char *q = p_;
        for (; *w; ++w, ++q) {
            if (q >= end_ || *q != *w) return false;
        }
        p_ = q;
        return true;
    }
    bool str(std::string *s) {
        if (p_ >= end_ || *p_ != '"') return false;
        ++p_;
        while (p_ < end_ && *p_ != '"') {
            if (*p_ == '\\') {
                ++p_;
                if (p_ >= end_) return false;
                const char e = *p_++;
                switch (e) {
                    case 'n': s->push_back('\n'); break;
                    case 't': s->push_back('\t'); break;
                    case 'r': s->push_back('\r'); break;
                    case 'b': s->push_back('\b'); break;
                    case 'f': s->push_back('\f'); break;
                    case 'u':
                        if (end_ - p_ < 4) return false;
                        p_ += 4;
                        s->push_back('?');
                        break;
                    default: s->push_back(e); break;
                }
            } else {
                s->push_back(*p_++);
            }
        }
        if (p_ >= end_) return false;
        ++p_;
        return true;
    }
    bool value(Value *v) {
        if (p_ >= end_) return false;
        const char c = *p_;
        if (c == '{') {
            ++p_;
            v->kind = Value::kObj;
            ws();
            if (p_ < end_ && *p_ == '}') {
                ++p_;
                return true;
            }
            for (;;) {
                ws();
                std::string k;
                if (!str(&k)) return false;
                ws();
                if (p_ >= end_ || *p_ != ':') return false;
                ++p_;
                ws();
                v->o.emplace_back(std::move(k), Value{});
                if (!value(&v->o.back().second)) return false;
                ws();
                if (p_ < end_ && *p_ == ',') {
                    ++p_;
                    continue;
                }
                if (p_ < end_ && *p_ == '}') {
                    ++p_;
                    return true;
                }
                return false;
            }
        }
        if (c == '[') {
            ++p_;
            v->kind = Value::kArr;
            ws();
            if (p_ < end_ && *p_ == ']') {
                ++p_;
                return true;
            }
            for (;;) {
                ws();
                v->a.emplace_back();
                if (!value(&v->a.back())) return false;
                ws();
                if (p_ < end_ && *p_ == ',') {
                    ++p_;
                    continue;
                }
                if (p_ < end_ && *p_ == ']') {
                    ++p_;
                    return true;
                }
                return false;
            }
        }
        if (c == '"') {
            v->kind = Value::kStr;
            return str(&v->s);
        }
        if (lit("true")) {
            v->kind = Value::kBool;
            v->b = true;
            return true;
        }
        if (lit("false")) {
            v->kind = Value::kBool;
            return true;
        }
        if (lit("null")) return true;
        // number
        const char *start = p_;
        bool integer = true;
        if (p_ < end_ && *p_ == '-') ++p_;
        while (p_ < end_ && ((*p_ >= '0' && *p_ <= '9') || *p_ == '.' || *p_ == 'e' || *p_ == 'E' || *p_ == '+' ||
                             *p_ == '-')) {
            if (*p_ == '.' || *p_ == 'e' || *p_ == 'E') integer = false;
            ++p_;
        }
        if (p_ == start) return false;
        const std::string text(start, p_);
        if (integer) {
            v->kind = Value::kInt;
            if (text[0] == '-') {
                v->i = std::strtoll(text.c_str(), nullptr, 10);
            } else {
                v->u = std::strtoull(text.c_str(), nullptr, 10);
                v->i = int64_t(v->u);
            }
            v->d = double(v->i);
        } else {
            v->kind = Value::kNum;
            v->d = std::strtod(text.c_str(), nullptr);
        }
        return true;
    }
};

inline bool parse(const std::string &text, Value *out) {
    Parser p(text.data(), text.size());
    return p.parse(out);
}

}  // namespace cere::testjson
