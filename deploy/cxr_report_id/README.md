# Laporan Foto Toraks Berbahasa Indonesia (MedGemma dan CheXOne + LoRA)

Paket ini berisi **dua model** yang menerima satu foto toraks (CXR) lalu menulis dua bagian laporan
radiologi dalam Bahasa Indonesia: **TEMUAN** (*findings*) dan **KESAN** (*impression*). Kedua model
dilatih dengan data, prompt, dan pengaturan yang sama, sehingga bisa ditampilkan berdampingan.

> **Model riset, bukan alat diagnosis.** Output-nya adalah draf yang wajib ditinjau dan disetujui
> dokter spesialis radiologi. Baca bagian [Keterbatasan](#keterbatasan) dan [Lisensi](#lisensi-dan-aturan-pakai)
> sebelum memasang model ini di aplikasi.

> **Di repositori GitHub, folder `models/` (adapter LoRA) tidak disertakan.** Adapter dilatih dari
> data MIMIC-CXR dan CheXpert Plus dan sebagian laporan training MIMIC-CXR terhafal, jadi adapter
> hanya boleh dibagikan sesuai lisensi dataset sumbernya.

| `model` | Base model | Ukuran | Hasil evaluasi | Lisensi |
|---|---|---|---|---|
| `medgemma` (default) | MedGemma 1.5 4B (Google, Gemma 3) | 4B | **Terbaik di rumah sakit yang tidak ikut training** (OpenI / IU X-ray). Disarankan sebagai model utama | HAI-DEF Terms of Use |
| `chexone` | CheXOne 3B (Stanford AIMI, Qwen2.5-VL) | 3B | **Terbaik di MIMIC-CXR**, juga di KESAN CheXpert, yaitu sumber data yang sudah dikenal CheXOne dari pretraining | CC BY-NC 4.0, **non-komersial** |

Rumah sakit Indonesia tidak pernah dilihat kedua model, jadi situasinya paling mirip dengan OpenI.
Cara model dibangun dan perbandingan lengkapnya ada di **[PIPELINE.md](PIPELINE.md)**.

## Isi folder

```
cxr_report_id/
├── README.md            dokumen ini: instalasi, cara pakai, integrasi ke aplikasi
├── PIPELINE.md          bagaimana pipeline dibangun + hasil evaluasi kedua model
├── NOTICE.txt           pemberitahuan lisensi kedua model, WAJIB ikut setiap kali folder ini dibagikan
├── requirements.txt     dependensi Python (versi yang sudah diuji)
├── selftest.py          cek setelah instalasi: environment, keutuhan file, VRAM dan waktu di mesinmu
├── inference.py         command line: gambar -> laporan (satu model atau semua)
├── serve_example.py     contoh HTTP API (FastAPI), bisa memilih model per request
├── cxr_report/          kode inference (di-import aplikasi)
│   ├── generator.py        ReportGenerator: load satu model + generate
│   ├── models.py           daftar model dan pengaturan masing-masing
│   ├── prompts.py          prompt bahasa Indonesia, persis seperti saat training
│   ├── preprocess.py       gambar -> 224x224 grayscale, seperti data training
│   └── postprocess.py      buang kalimat berulang dari output
└── models/
    ├── medgemma/        adapter LoRA MedGemma (±265 MB) + tokenizer/processor
    └── chexone/         adapter LoRA CheXOne (±240 MB) + tokenizer/processor
```

Folder ini **tidak** berisi bobot base model. MedGemma (±8,6 GB) dan CheXOne (±8,2 GB) diunduh
otomatis dari Hugging Face saat model pertama kali dijalankan (lihat Instalasi).

## Kebutuhan

| | |
|---|---|
| Python | 3.11 atau 3.12 |
| GPU (disarankan) | NVIDIA dengan dukungan bfloat16 (RTX 30xx ke atas, A10, A100, L4, ...). Dilatih dan diuji di RTX 5060 Ti 16 GB. VRAM (4-bit, default), **perkiraan**: MedGemma ±4 GB, CheXOne ±3 GB setelah dimuat, ditambah ±1–2 GB saat batch 8 gambar. Kedua model sekaligus: GPU 12 GB cukup, 16 GB lebih lega. Satu model saja: GPU 8 GB. Angka pasti di GPU-mu: `python selftest.py` |
| CPU saja | Bisa (`--cpu`), tapi lambat (hitungan menit per gambar) dan butuh RAM ±10 GB per model (perkiraan) |
| Disk | ±17 GB untuk kedua base model di cache Hugging Face, ditambah ±0,5 GB adapter |
| Akun Hugging Face | MedGemma bersifat *gated*: setujui ketentuannya di <https://huggingface.co/google/medgemma-1.5-4b-it>, lalu login. CheXOne tidak *gated* |

## Instalasi

```bash
python -m venv .venv
.venv\Scripts\activate                # Linux/macOS: source .venv/bin/activate

# 1. PyTorch sesuai GPU/CUDA (contoh: CUDA 12.8; untuk CPU saja ganti cu128 -> cpu)
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
# 2. sisanya
pip install -r requirements.txt
# 3. login Hugging Face (akun yang sudah menyetujui ketentuan MedGemma), atau set env HF_TOKEN
hf auth login
# 4. cek semuanya jalan (unduh base model saat pertama kali, lalu ukur VRAM dan waktu)
python selftest.py                              # gambar sintetis: hanya membuktikan model termuat dan menulis
python selftest.py --image cxr.png --view PA    # foto toraks asli: sekaligus menampilkan kedua laporan
```

`selftest.py` harus diakhiri `SELFTEST PASSED`. Ia juga mengecek sha256 adapter (file rusak saat
disalin atau di-zip akan ketahuan) dan mencetak tabel VRAM dan waktu per model untuk mesin itu.

Saat pertama dijalankan, base model diunduh ke cache Hugging Face, lalu dipakai ulang tanpa unduh
lagi.

**Server tanpa internet:** unduh sekali di mesin lain, salin foldernya ke server, lalu tunjuk
foldernya:

```bash
hf download google/medgemma-1.5-4b-it --local-dir medgemma-1.5-4b-it
hf download StanfordAIMI/CheXOne --local-dir CheXOne
python inference.py --image cxr.png --view PA --model medgemma --base_model medgemma-1.5-4b-it
python inference.py --image cxr.png --view PA --model chexone --base_model CheXOne
```

## Cara pakai

### Command line

```bash
python inference.py --image cxr.png --view PA                   # MedGemma (default)
python inference.py --image cxr.png --view PA --model chexone
python inference.py --image cxr.png --view PA --model all       # kedua model, bergantian
python inference.py --image a.png b.png --view PA LATERAL --model all --json
python inference.py --image cxr.png --cpu                       # tanpa GPU (lambat)
```

Dengan `--model all`, model dimuat satu per satu dan dilepas dari GPU sebelum model berikutnya,
jadi cukup dengan VRAM satu model.

### Dari kode Python (cara yang disarankan untuk aplikasi)

```python
from cxr_report import ReportGenerator

# load SEKALI saat aplikasi start (±1 menit per model)
models = {"medgemma": ReportGenerator("medgemma"), "chexone": ReportGenerator("chexone")}

for name, gen in models.items():
    report = gen.generate("cxr.png", view="PA")
    print(name, report["findings"])          # TEMUAN
    print(name, report["impression"])        # KESAN

# beberapa gambar sekaligus (lebih efisien di GPU; maksimal ±8 per batch untuk GPU 16 GB)
reports = models["medgemma"].generate_batch(["a.png", "b.png"], views=["PA", "LATERAL"])
```

`image` boleh berupa path file, `bytes` (misalnya dari upload), file object, numpy array, atau
`PIL.Image`. Opsi `ReportGenerator(model, ...)`: `base_model` (folder lokal base model), `device`
(`"cuda"`, `"cuda:1"`, `"cpu"`), `load_in_4bit` (default `True` di GPU). Daftar model ada di
`cxr_report.MODELS`.

### HTTP API (contoh)

```bash
pip install fastapi==0.115.12 uvicorn==0.34.2 python-multipart==0.0.20
uvicorn serve_example:app --host 0.0.0.0 --port 8000      # memuat kedua model
# satu model saja: set CXR_MODELS=medgemma   (Linux/macOS: CXR_MODELS=medgemma uvicorn ...)

curl http://localhost:8000/models
curl -X POST http://localhost:8000/report -F "file=@cxr.png" -F "view=PA" -F "model=chexone"
curl -X POST http://localhost:8000/report -F "file=@cxr.png" -F "view=PA" -F "model=all"
```

Respons untuk satu model (`model` kosong = model pertama yang dimuat, yaitu MedGemma):

```json
{
  "model": "chexone",
  "findings": "...",
  "impression": "...",
  "view": "PA",
  "seconds": 9.8,
  "disclaimer": "Draf laporan otomatis dari model riset. Bukan diagnosis; ..."
}
```

Dengan `model=all`, laporannya ada di `"reports": {"medgemma": {...}, "chexone": {...}}`.
`GET /health` mengembalikan status dan konfigurasi setiap model. Ini hanya contoh minimal: belum
ada autentikasi, batas ukuran upload, ataupun antrean.

## Input

- **Gambar:** foto toraks dalam format PNG atau JPEG, grayscale atau RGB, 8 atau 16 bit. Gambar
  otomatis diubah ke 224×224 grayscale, sama seperti data training: seluruh gambar di-resize
  tanpa crop dan tanpa padding. Setelah itu MedGemma memperbesarnya ke 896×896 dan CheXOne ke
  448×448, juga sama seperti saat training.
- **DICOM belum didukung.** Export dulu ke PNG dengan window yang tersimpan di file DICOM.
  Gambar MONOCHROME1 harus dibalik (inverted) agar tulang tampak putih.
- **`view` (proyeksi):** `AP`, `PA`, `LATERAL` (`LL` dan `RL` juga diterima), atau kosong/`None`
  kalau tidak diketahui. Huruf besar/kecil tidak berpengaruh. Isi kalau tahu, karena model dilatih
  dengan informasi ini; nilai lain ditolak dengan `ValueError`.
- Model **tidak bisa** mengenali gambar yang bukan foto toraks. Gambar apa pun tetap akan dibuatkan
  "laporan" yang kedengarannya meyakinkan, jadi aplikasi perlu memastikan input-nya memang CXR.

## Output

- Dua teks per model: `findings` (TEMUAN) dan `impression` (KESAN), dihasilkan dengan dua kali
  generate.
- Teks **huruf kecil semua**, karena data training dinormalisasi ke huruf kecil. Kalau perlu,
  aplikasi bisa mengubah huruf pertama tiap kalimat menjadi kapital.
- KESAN kadang berupa daftar bernomor (`1. ... 2. ...`).
- Kalimat yang berulang sudah dibuang (`postprocess.py`). `return_raw=True` (atau `--raw` di
  command line) juga mengembalikan output asli model.
- Decoding greedy (tanpa sampling), jadi input yang sama memberi output yang sama, kecuali
  kadang ada perbedaan kecil kalau gambar diproses dalam batch yang berbeda.

## Integrasi ke aplikasi: tips

- **Load model satu kali** saat aplikasi start. Jangan load per request.
- **Menampilkan dua model:**
  - beri label nama model di setiap laporan;
  - tampilkan berdampingan sebagai dua draf;
  - perbedaan isi antara kedua draf bisa menjadi tanda bagi radiolog untuk memeriksa bagian itu
    lebih teliti.
- **Satu GPU memproses satu request dalam satu waktu.** `serve_example.py` memakai lock untuk
  itu. Dengan `model=all`, kedua model dijalankan bergantian, jadi waktunya kira-kira dua kali
  lipat. Kalau request banyak, pakai antrean (queue) atau `generate_batch`.
- **Waktu** (RTX 5060 Ti 16 GB, 4-bit), diukur untuk MedGemma pada versi paket sebelumnya. Untuk
  kedua model di mesinmu, lihat tabel di akhir `selftest.py`:
  - memuat model: ±30 detik, setelah base model ada di cache;
  - satu gambar (TEMUAN + KESAN): ±14 detik, diukur pada request pertama;
  - batch 8 gambar: ±3 detik per gambar. CheXOne di pipeline evaluasi ±30% lebih cepat per gambar
    daripada MedGemma.

  Karena satu request makan belasan detik, sebaiknya laporan diproses secara asinkron di sisi
  aplikasi, misalnya dengan status "sedang diproses".
- **Catat versi model** di log setiap laporan (sha256 `adapter_model.safetensors`, lihat
  [Versi](#versi)).
- Tampilkan laporan sebagai **draf** yang harus disunting dan disetujui radiolog. Jangan pernah
  menampilkannya sebagai hasil akhir.

## Keterbatasan

Dirangkum dari evaluasi (detail di [PIPELINE.md](PIPELINE.md)):

- **Menyebut "foto sebelumnya" yang tidak ada.** Model hanya melihat satu gambar, tapi sering
  menulis "stabil", "persisten", atau "dibandingkan dengan studi sebelumnya". Penyebabnya, sekitar
  59% KESAN di data training CheXpert merujuk ke foto terdahulu. Kadang kedua model juga menulis
  tanggal pemeriksaan yang dikarang.
- **Kalimat template.** Banyak output memakai kalimat yang sama untuk gambar yang berbeda, terutama
  untuk foto normal.
- **CheXOne terbawa gaya laporan MIMIC-CXR** di rumah sakit lain: TEMUAN-nya lebih panjang dan sering
  diawali kalimat prosedur ("foto PA dan proyeksi lateral diperoleh ...").
- **Data training dari rumah sakit Amerika Serikat** (CheXpert: Stanford; MIMIC-CXR: Boston),
  dengan laporan yang diterjemahkan ke Bahasa Indonesia. Model belum diuji pada foto maupun gaya
  laporan rumah sakit di Indonesia.
- **Belum ada validasi klinis.** Skor metrik teks sedang. Model bisa melewatkan temuan atau
  mengarang temuan yang tidak ada.
- Dilatih untuk foto toraks orang dewasa (sebagian besar AP dan PA, sebagian lateral), dengan
  resolusi 224×224, sehingga temuan yang sangat kecil bisa terlewat.

## Lisensi dan aturan pakai

- **MedGemma (`models/medgemma/`): Health AI Developer Foundations (HAI-DEF) Terms of Use.** Adapter
  ini adalah *Model Derivative* dari MedGemma, jadi:
  - `NOTICE.txt` wajib ikut setiap kali folder ini dibagikan;
  - setiap penerima harus mendapat ketentuannya: <https://developers.google.com/health-ai-developer-foundations/terms>;
  - dilarang dipakai untuk hal-hal di *Prohibited Use Policy* HAI-DEF;
  - **penggunaan klinis** (diagnosis atau terapi pasien, termasuk dalam studi riset) memerlukan
    izin dari regulator kesehatan yang berwenang.
- **CheXOne (`models/chexone/`): CC BY-NC 4.0, hanya untuk penggunaan non-komersial.**
  - Atribusi ke Stanford AIMI wajib dicantumkan; teksnya ada di `NOTICE.txt`.
  - CheXOne sendiri dibangun dari Qwen2.5-VL-3B-Instruct, yang punya ketentuan lisensi sendiri.
- **Data training:**
  - CheXpert Plus: *Research Use Agreement* dari Stanford AIMI, hanya untuk riset non-komersial;
  - MIMIC-CXR: *PhysioNet Credentialed Health Data License 1.5.0*;
  - OpenI dipakai hanya untuk pengujian.

  Paket ini disiapkan untuk riset dan demo non-komersial. Sebelum dipakai di luar itu, cek dulu
  dengan dosen pembimbing.

## Versi

| | MedGemma | CheXOne |
|---|---|---|
| Run | `medgemma_full`, step 16072 (selesai 2026-10-14) | `chexone_full`, step 16072 (selesai 2026-10-17) |
| Base model | `google/medgemma-1.5-4b-it` | `StanfordAIMI/CheXOne` |
| Training | Unsloth QLoRA 4-bit, LoRA r=32 di language model | sama persis |
| sha256 `adapter_model.safetensors` | `a387a1f0a0cd00a386023347e704c66b7dbb16fc3b82a745001ce7f99bf5487e` | `9c113d125061abcfdcb688aae9d69dc4cf09d648a03bf819af5bc0e6d3637fbc` |

- **Diuji dengan:** Python 3.12, torch 2.11.0, transformers 5.5.0, peft 0.21.2, bitsandbytes 0.50.2.
- **Pengecekan tanpa GPU (lolos):**
  - input model (token prompt, gambar) identik dengan pipeline evaluasi untuk kedua model;
  - lapisan yang dikompres 4-bit sama persis dengan saat training;
  - struktur adapter cocok (MedGemma 476/476, CheXOne 504/504 tensor LoRA);
  - API HTTP diuji dengan model tiruan.
- **Kesetaraan hasil di GPU:**
  - Versi paket sebelumnya (MedGemma saja) berbeda paling banyak ±0,01 dari pipeline evaluasi.
  - Versi ini memperbaiki satu perbedaan: vision encoder MedGemma kini tetap 16-bit seperti saat
    training (sebelumnya ikut dikompres 4-bit).
  - Versi ini belum diuji ulang di GPU oleh tim riset; perbaikannya sudah diverifikasi lapisan per
    lapisan tanpa GPU. Di mesin aplikasi, `selftest.py` memastikan kedua model termuat dan menulis laporan.
