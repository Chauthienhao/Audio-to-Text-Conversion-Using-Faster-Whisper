"""Chuẩn bị dữ liệu VIVOS (tập test) trên Colab: tải từ Hugging Face, giải nén riêng phần test,
tạo manifest đầy đủ (760 câu) và manifest 200 câu cố định (seed 42, trùng với máy cục bộ)."""
import csv, shutil, tarfile
from pathlib import Path

HF_REPO = "AILAB-VNUHCM/vivos"
HF_TAR = "data/vivos.tar.gz"           # chứa cả train và test (~1.4 GB), chỉ giải nén test

# 200 câu lấy bằng pandas .sample(n=200, random_state=42) từ 760 câu test, sắp theo đường dẫn.
EVAL_IDS = [
    "VIVOSDEV01_R003", "VIVOSDEV01_R027", "VIVOSDEV01_R071", "VIVOSDEV01_R078",
    "VIVOSDEV01_R117", "VIVOSDEV01_R139", "VIVOSDEV01_R146", "VIVOSDEV01_R170",
    "VIVOSDEV01_T043", "VIVOSDEV02_R009", "VIVOSDEV02_R069", "VIVOSDEV02_R089",
    "VIVOSDEV02_R099", "VIVOSDEV02_R130", "VIVOSDEV02_R149", "VIVOSDEV02_R154",
    "VIVOSDEV02_R170", "VIVOSDEV02_T013", "VIVOSDEV02_T050", "VIVOSDEV03_R007",
    "VIVOSDEV03_R010", "VIVOSDEV03_R041", "VIVOSDEV03_R046", "VIVOSDEV03_R056",
    "VIVOSDEV03_R082", "VIVOSDEV03_R119", "VIVOSDEV03_R144", "VIVOSDEV03_R163",
    "VIVOSDEV03_R180", "VIVOSDEV03_R182", "VIVOSDEV03_R198", "VIVOSDEV03_T009",
    "VIVOSDEV03_T017", "VIVOSDEV03_T018", "VIVOSDEV03_T041", "VIVOSDEV03_T047",
    "VIVOSDEV03_T048", "VIVOSDEV04_R002", "VIVOSDEV04_R023", "VIVOSDEV04_R086",
    "VIVOSDEV04_R092", "VIVOSDEV04_R100", "VIVOSDEV04_R162", "VIVOSDEV04_R182",
    "VIVOSDEV04_R186", "VIVOSDEV04_R189", "VIVOSDEV04_T032", "VIVOSDEV04_T044",
    "VIVOSDEV05_055", "VIVOSDEV05_065", "VIVOSDEV05_125", "VIVOSDEV05_170",
    "VIVOSDEV05_238", "VIVOSDEV05_249", "VIVOSDEV05_263", "VIVOSDEV05_295",
    "VIVOSDEV06_022", "VIVOSDEV06_042", "VIVOSDEV06_061", "VIVOSDEV06_070",
    "VIVOSDEV06_078", "VIVOSDEV06_079", "VIVOSDEV06_103", "VIVOSDEV06_126",
    "VIVOSDEV06_143", "VIVOSDEV06_222", "VIVOSDEV06_248", "VIVOSDEV06_261",
    "VIVOSDEV06_272", "VIVOSDEV07_039", "VIVOSDEV07_064", "VIVOSDEV07_084",
    "VIVOSDEV07_141", "VIVOSDEV07_185", "VIVOSDEV07_190", "VIVOSDEV08_005",
    "VIVOSDEV08_085", "VIVOSDEV08_090", "VIVOSDEV08_093", "VIVOSDEV08_129",
    "VIVOSDEV08_166", "VIVOSDEV08_196", "VIVOSDEV08_210", "VIVOSDEV08_222",
    "VIVOSDEV08_270", "VIVOSDEV09_038", "VIVOSDEV09_055", "VIVOSDEV09_083",
    "VIVOSDEV09_091", "VIVOSDEV09_098", "VIVOSDEV09_134", "VIVOSDEV09_142",
    "VIVOSDEV09_158", "VIVOSDEV09_163", "VIVOSDEV09_194", "VIVOSDEV09_222",
    "VIVOSDEV09_246", "VIVOSDEV09_267", "VIVOSDEV09_286", "VIVOSDEV09_288",
    "VIVOSDEV09_289", "VIVOSDEV10_010", "VIVOSDEV10_014", "VIVOSDEV10_020",
    "VIVOSDEV10_043", "VIVOSDEV10_045", "VIVOSDEV10_051", "VIVOSDEV10_162",
    "VIVOSDEV10_172", "VIVOSDEV10_212", "VIVOSDEV10_257", "VIVOSDEV11_075",
    "VIVOSDEV11_123", "VIVOSDEV11_136", "VIVOSDEV11_200", "VIVOSDEV11_234",
    "VIVOSDEV11_269", "VIVOSDEV11_298", "VIVOSDEV12_032", "VIVOSDEV12_066",
    "VIVOSDEV12_068", "VIVOSDEV12_079", "VIVOSDEV12_116", "VIVOSDEV12_166",
    "VIVOSDEV12_197", "VIVOSDEV12_223", "VIVOSDEV12_232", "VIVOSDEV12_275",
    "VIVOSDEV12_280", "VIVOSDEV13_054", "VIVOSDEV13_078", "VIVOSDEV13_139",
    "VIVOSDEV13_147", "VIVOSDEV13_150", "VIVOSDEV13_225", "VIVOSDEV13_241",
    "VIVOSDEV13_255", "VIVOSDEV13_263", "VIVOSDEV13_267", "VIVOSDEV14_027",
    "VIVOSDEV14_093", "VIVOSDEV14_145", "VIVOSDEV14_163", "VIVOSDEV14_180",
    "VIVOSDEV14_199", "VIVOSDEV14_225", "VIVOSDEV14_235", "VIVOSDEV14_291",
    "VIVOSDEV15_038", "VIVOSDEV15_092", "VIVOSDEV15_117", "VIVOSDEV15_168",
    "VIVOSDEV15_180", "VIVOSDEV15_203", "VIVOSDEV15_208", "VIVOSDEV15_216",
    "VIVOSDEV15_239", "VIVOSDEV15_267", "VIVOSDEV15_299", "VIVOSDEV16_033",
    "VIVOSDEV16_067", "VIVOSDEV16_068", "VIVOSDEV16_153", "VIVOSDEV16_175",
    "VIVOSDEV16_193", "VIVOSDEV16_228", "VIVOSDEV16_239", "VIVOSDEV16_247",
    "VIVOSDEV17_099", "VIVOSDEV17_107", "VIVOSDEV17_120", "VIVOSDEV17_147",
    "VIVOSDEV17_169", "VIVOSDEV17_170", "VIVOSDEV17_204", "VIVOSDEV17_217",
    "VIVOSDEV17_297", "VIVOSDEV18_021", "VIVOSDEV18_051", "VIVOSDEV18_064",
    "VIVOSDEV18_118", "VIVOSDEV18_120", "VIVOSDEV18_183", "VIVOSDEV18_240",
    "VIVOSDEV18_259", "VIVOSDEV18_263", "VIVOSDEV18_265", "VIVOSDEV18_280",
    "VIVOSDEV18_282", "VIVOSDEV18_293", "VIVOSDEV19_008", "VIVOSDEV19_022",
    "VIVOSDEV19_029", "VIVOSDEV19_030", "VIVOSDEV19_111", "VIVOSDEV19_133",
    "VIVOSDEV19_196", "VIVOSDEV19_259", "VIVOSDEV19_262", "VIVOSDEV19_274",
]


def download_and_extract_test(data_dir):
    """Tải vivos.tar.gz rồi chỉ ghi các file nằm trong thư mục test/ vào data_dir/vivos/test/."""
    from huggingface_hub import hf_hub_download
    data_dir = Path(data_dir)
    test_dir = data_dir / "vivos" / "test"
    if (test_dir / "prompts.txt").is_file() and any((test_dir / "waves").glob("*/*.wav")):
        print("Đã có dữ liệu test, bỏ qua tải.")
        return test_dir
    print("Đang tải", HF_TAR, "(khoảng 1.4 GB) ...")
    tar_path = hf_hub_download(repo_id=HF_REPO, filename=HF_TAR, repo_type="dataset")
    n = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if "test" not in parts:
                continue
            rel = Path(*parts[parts.index("test"):])       # tự dựng đường dẫn đích, không dùng tên trong tar
            if ".." in rel.parts:
                continue
            dest = test_dir.parent / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            n += 1
    print(f"Đã giải nén {n} file test vào {test_dir}")
    return test_dir


def read_prompts(test_dir):
    prompts = {}
    for line in open(Path(test_dir) / "prompts.txt", encoding="utf-8"):
        key, _, text = line.rstrip("\n").partition(" ")
        if key:
            prompts[key] = text.strip()
    return prompts


def _write_manifest(path, ids, prompts, test_dir):
    rows = []
    for i in ids:
        wav = Path(test_dir) / "waves" / i.split("_")[0] / f"{i}.wav"
        if not wav.is_file():
            raise FileNotFoundError(wav)
        rows.append((str(wav), prompts[i]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["audio", "reference"]); w.writerows(rows)
    return len(rows)


def build_manifests(data_dir):
    """Trả về (đường dẫn manifest 760 câu, đường dẫn manifest 200 câu)."""
    data_dir = Path(data_dir)
    test_dir = data_dir / "vivos" / "test"
    prompts = read_prompts(test_dir)
    full, sub = data_dir / "manifest.csv", data_dir / "manifest_eval.csv"
    n1 = _write_manifest(full, list(prompts), prompts, test_dir)
    n2 = _write_manifest(sub, EVAL_IDS, prompts, test_dir)
    print(f"manifest.csv: {n1} câu | manifest_eval.csv: {n2} câu")
    return full, sub
