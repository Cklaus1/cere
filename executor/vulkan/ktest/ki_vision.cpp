// Inkling S12: the vision tower on Vulkan (src/device/inkling/ki_vision.h, inkling design note 3.6, S7's gate),
// standalone: it loads the S0 artifact's 422 `model.visual.*` tensors (exact checkpoint bytes), encodes the recorded
// images and writes the merged features for executor/vulkan/tools/ki_vision_compare.py.
//
// ki_vision SPV_DIR PACK_DIR IN_DIR OUT_DIR NAME... [--k2]
//   IN_DIR/NAME.pix: the processor's patch rows, BF16 [h * w][1,200]; IN_DIR/NAME.grid: int32 h, w (20-px patches)
//   (ki_vision_compare.py prep writes both). OUT_DIR/NAME.CASE.bf16: BF16 [h * w / 4][2,304] per case:
//   single (each image alone), batched (all in one call), no_class / swap_rope (G7 image fail-first mutants, batched).
//   A second batched call must be bit-identical to the first (printed REPEAT).
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "device/inkling/ki_vision.h"
#include "ki_pack.h"
#include "launch/vk_launch.h"
#include "launch/vk_launch_testing.h"

namespace vk = cere::vk;
namespace vkink = cere::vkink;
using kipack::die;

namespace {

struct Image {
    std::string name;
    vkink::VisionGrid grid;
    std::vector<uint16_t> pix;
};

std::vector<uint8_t> read_file(const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) die("cannot open " + path);
    std::vector<uint8_t> v;
    uint8_t buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + n);
    std::fclose(f);
    return v;
}

void write_file(const std::string &path, const void *p, size_t bytes) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f || std::fwrite(p, 1, bytes, f) != bytes) die("cannot write " + path);
    std::fclose(f);
}

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Encodes `imgs` in one call; returns the output rows (BF16 [tokens][2,304]).
std::vector<uint16_t> encode(vk::Context &ctx, vkink::Vision &v, const std::vector<const Image *> &imgs) {
    std::vector<vkink::VisionGrid> grids;
    uint64_t rows = 0;
    for (const Image *im : imgs) {
        std::memcpy(v.pixels_host() + rows * vkink::kPatchRow, im->pix.data(), im->pix.size() * 2);
        rows += im->pix.size() / vkink::kPatchRow;
        grids.push_back(im->grid);
    }
    const double t0 = now_s();
    if (const cere_status st = v.encode(grids.data(), uint32_t(grids.size())); st != CERE_OK)
        die("encode (" + std::to_string(int(st)) + "): " + ctx.last_error());
    std::printf("encode %zu image(s), %llu patches: %.3f s\n", imgs.size(), (unsigned long long)rows, now_s() - t0);
    return std::vector<uint16_t>(v.out_host(), v.out_host() + rows / 4 * vkink::kVisOut);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 6) {
        std::printf("usage: ki_vision SPV_DIR PACK_DIR IN_DIR OUT_DIR NAME... [--k2]\n");
        return 2;
    }
    bool k2 = false;
    std::vector<Image> images;
    uint64_t patches = 0;
    for (int i = 5; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--k2") {
            k2 = true;
            continue;
        }
        Image im;
        im.name = a;
        const std::vector<uint8_t> g = read_file(std::string(argv[3]) + "/" + a + ".grid");
        if (g.size() != 8) die(a + ".grid: expected two int32");
        std::memcpy(&im.grid.h, g.data(), 4);
        std::memcpy(&im.grid.w, g.data() + 4, 4);
        const std::vector<uint8_t> p = read_file(std::string(argv[3]) + "/" + a + ".pix");
        if (p.size() != uint64_t(im.grid.h) * im.grid.w * vkink::kPatchRow * 2) die(a + ".pix: size differs from grid");
        im.pix.resize(p.size() / 2);
        std::memcpy(im.pix.data(), p.data(), p.size());
        patches += uint64_t(im.grid.h) * im.grid.w;
        images.push_back(std::move(im));
    }
    if (images.empty() || images.size() > vkink::kVisMaxImages) die("1 .. 32 images");

    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    auto ctx = vk::testing::create_or_die(argv[1]);
    std::printf("device: %s\n", ctx->info().name.c_str());
    kipack::Pack pk;
    pk.open(std::string(argv[2]) + "/model-00001-of-00001.safetensors");
    kipack::Weights w{*ctx, pk};
    const double t0 = now_s();
    for (uint32_t i = 0; i < vkink::kVisTensors; ++i) w.load(vkink::vision_tensor_name(i));
    std::printf("weights: %.3f GB in %.1f s\n", double(w.total) / 1e9, now_s() - t0);

    const std::string out = argv[4];
    std::vector<const Image *> all;
    for (const Image &im : images) all.push_back(&im);
    const auto run = [&](const char *kase, bool no_class, bool swap_rope) {
        vkink::Vision v(*ctx);
        vkink::Vision::Options opt;
        opt.max_patches = uint32_t((patches + 3) / 4 * 4);
        opt.k2 = k2;
        opt.no_class = no_class;
        opt.swap_rope = swap_rope;
        std::string why;
        if (const cere_status st = v.init(w.find(), opt, &why); st != CERE_OK)
            die("init (" + std::to_string(int(st)) + "): " + why + " " + ctx->last_error());
        const auto save = [&](const std::vector<uint16_t> &rows, const char *c) {
            uint64_t at = 0;
            for (const Image &im : images) {
                const uint64_t n = uint64_t(im.grid.h) * im.grid.w / 4 * vkink::kVisOut;
                write_file(out + "/" + im.name + "." + c + ".bf16", rows.data() + at, n * 2);
                at += n;
            }
        };
        if (std::strcmp(kase, "batched") == 0) {
            for (const Image &im : images) {
                const std::vector<uint16_t> r = encode(*ctx, v, {&im});
                write_file(out + "/" + im.name + ".single.bf16", r.data(), r.size() * 2);
            }
            const std::vector<uint16_t> first = encode(*ctx, v, all);
            const std::vector<uint16_t> again = encode(*ctx, v, all);
            std::printf("REPEAT %s\n", first == again ? "bit-identical" : "DIFFER");
            save(first, kase);
            if (first != again) std::exit(1);
        } else {
            save(encode(*ctx, v, all), kase);
        }
    };
    run("batched", false, false);
    run("no_class", true, false);
    run("swap_rope", false, true);
    return 0;
}
