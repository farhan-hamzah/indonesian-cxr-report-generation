# Metodologi Fine-Tuning: MedGemma-1.5-4B dan CheXOne untuk Laporan CXR Bahasa Indonesia

Dokumen ini menjelaskan **apa yang sebenarnya terjadi** di dalam kedua pendekatan fine-tuning
(`medgemma_full` dan `chexone_full`), dari teks terjemahan sampai laporan yang keluar dari model,
supaya prosesnya tidak terasa *black box*. Semua angka diambil dari kode, file adapter hasil
training, `trainer_state.json`, dan output `tools/check_model.py` (dicek 2026-10-20).

Kedua model memakai **pendekatan yang sama** (fine-tuning end-to-end VLM dengan QLoRA), hanya
*base model*-nya yang berbeda. Jadi bagian 1–8 berlaku untuk keduanya; perbedaannya dirangkum
di bagian 9.

---

## Daftar isi

0. [Ringkasan satu halaman](#0-ringkasan-satu-halaman)
1. [Formulasi tugas: apa yang dipelajari model](#1-formulasi-tugas-apa-yang-dipelajari-model)
2. [Data: dari laporan Inggris sampai sampel training](#2-data-dari-laporan-inggris-sampai-sampel-training)
3. [Arsitektur kedua model (white-box)](#3-arsitektur-kedua-model-white-box)
4. [Prompt dan chat template](#4-prompt-dan-chat-template)
5. [Loss dan prompt masking](#5-loss-dan-prompt-masking)
6. [QLoRA: bagian mana yang dilatih, bagian mana yang dibekukan](#6-qlora-bagian-mana-yang-dilatih-bagian-mana-yang-dibekukan)
7. [Optimisasi dan jalannya training](#7-optimisasi-dan-jalannya-training)
8. [Inferensi dan post-processing](#8-inferensi-dan-post-processing)
9. [Perbandingan MedGemma vs CheXOne (side by side)](#9-perbandingan-medgemma-vs-chexone-side-by-side)
10. [Evaluasi (ringkas)](#10-evaluasi-ringkas)
11. [Alasan di balik pilihan desain](#11-alasan-di-balik-pilihan-desain)
12. [Yang tidak dilakukan / keterbatasan](#12-yang-tidak-dilakukan--keterbatasan)
13. [Peta kode dan cara reproduksi](#13-peta-kode-dan-cara-reproduksi)
14. [Catatan: pendekatan ketiga (rrg_qwen)](#14-catatan-pendekatan-ketiga-rrg_qwen)

---

## 0. Ringkasan satu halaman

```
                      ┌──────────────────────── DIBEKUKAN ────────────────────────┐   ┌──── DILATIH ────┐
foto CXR 224x224 ──►  resize ──► vision encoder ──► projector ──► 256 token visual ─┐
(grayscale → RGB)                                                                   ├─► decoder bahasa ──► TEMUAN / KESAN
prompt Indonesia ──►  tokenizer ──────────────────────────────────► token teks ─────┘   (bobot 4-bit beku     (bahasa Indonesia)
                                                                                         + adapter LoRA r=32)
```

| | MedGemma-1.5-4B (`medgemma_full`) | CheXOne (`chexone_full`) |
|---|---|---|
| Base model | `google/medgemma-1.5-4b-it` (Gemma 3) | `StanfordAIMI/CheXOne` (Qwen2.5-VL-3B + post-training CXR) |
| Sifat | generalis medis, multibahasa | spesialis CXR, bahasa Inggris |
| Total parameter base | 4,30 miliar | ±3,75 miliar |
| Parameter yang dilatih (LoRA) | 59.604.992 (1,39%) | 59.867.136 (±1,6%) |
| Yang dilatih | LoRA di q/k/v/o/gate/up/down **semua layer decoder bahasa** | sama |
| Yang beku | vision encoder, projector, embedding, bobot asli decoder | sama |
| Data training | 257.138 sampel (gambar + tugas; sampel MIMIC diulang 4×), urutan identik | identik |
| Hyper-parameter | lr 2e-4 cosine, batch efektif 16, 1 epoch = 16.072 step | identik |
| Waktu training (RTX 5060 Ti 16 GB) | ±18,8 s/step, ±84 jam | ±8,0 s/step, ±36 jam |
| Eval loss akhir (dev) | 1,2098 | 0,9370 (tidak bisa dibandingkan, tokenizer beda) |

Intinya: **model belajar menulis laporan bahasa Indonesia dari fitur visual yang sudah ada**.
"Mata" model (vision encoder) tidak diubah; yang diubah hanya "cara menulis" (decoder bahasa),
lewat matriks LoRA kecil yang ditambahkan di atas bobot asli yang dibekukan.

---

## 1. Formulasi tugas: apa yang dipelajari model

Setiap **sampel training = 1 gambar + 1 tugas**:

- tugas `findings` → model harus menulis bagian **TEMUAN**
- tugas `impression` → model harus menulis bagian **KESAN**

Satu gambar bisa menghasilkan 1 atau 2 sampel, tergantung bagian referensi mana yang ada. Kedua
tugas dilatih dengan **satu adapter yang sama**; model tahu bagian mana yang diminta dari kalimat
prompt (`... bagian TEMUAN ...` atau `... bagian KESAN ...`).

Secara matematis, model mempelajari distribusi bersyarat

```
P(laporan | gambar, prompt) = Π_t  P(token_t | gambar, prompt, token_<t)
```

dan dilatih dengan *next-token prediction* (cross-entropy) hanya pada token laporan (lihat bagian 5).

Yang **tidak** ada di pendekatan ini: tidak ada classifier terpisah, tidak ada retrieval, tidak ada
tahap alignment. Model end-to-end: gambar masuk, teks keluar. (Opsi `--hints` untuk memasukkan
prediksi classifier ke prompt ada di kode, tetapi **tidak dipakai** di kedua run.)

---

## 2. Data: dari laporan Inggris sampai sampel training

### 2.1 Sumber dan terjemahan

| Dataset | Bagian yang dipakai | Peran |
|---|---|---|
| CheXpert Plus | split *train* resmi → dibagi per pasien 85% / 15% | train / dev |
| CheXpert Plus | split *valid* resmi (234 gambar, 200 pasien) | test |
| MIMIC-CXR | split *validate* resmi (2.852 gambar) | train (tambahan) |
| MIMIC-CXR | split *test* resmi (4.667 gambar) | test |
| IU-Xray / OpenI | seluruhnya (7.430 gambar) | test eksternal |

Bagian *findings* dan *impression* laporan bahasa Inggris diterjemahkan ke bahasa Indonesia dengan
**TranslateGemma-4B** (dilakukan tim sebelum pipeline ini; file ada di
`dataset/bahasaIndonesiaText/`). Untuk CheXpert Plus, hasil terjemahan berbentuk
`"temuan: ... kesan: ..."` lalu dipecah lagi dengan regex di `prepare_data.py`; satu bagian
terjemahan hanya dipakai kalau bagian bahasa Inggris aslinya memang ada (baris yang struktur
`temuan/kesan`-nya rusak saat diterjemahkan otomatis terbuang).

Pembagian train/dev CheXpert diambil dari pipeline bahasa Inggris (`rrg_pipeline`), jadi dev/test
identik dengan baseline Inggris. Cek kebocoran: **0** pasien/studi/path yang sama antara train dan
dev/test, **0** file gambar identik (md5) antara train dan test.

### 2.2 Normalisasi teks (`rrg/text.py`)

**`clean_text`** dipakai untuk **semua** teks (target training *dan* referensi test):

1. buang awalan `temuan:` / `kesan:`
2. huruf kecil semua (*lowercase*)
3. kamus istilah 10 entri untuk memperbaiki terjemahan yang sering salah:

   | hasil terjemahan | diganti menjadi |
   |---|---|
   | kekeruhan | opasitas |
   | wilayah udara | ruang udara |
   | penumpulan | blunting |
   | sudut costofrenum / sudut kosto | sudut kostofrenikus |
   | tabung nasogastrik | selang nasogastrik |
   | tabung endotrakeal | selang endotrakeal |
   | kabel alat pacu jantung | kawat alat pacu jantung |
   | tidak ada kelainan akut / tidak ada temuan akut | tidak tampak kelainan akut |

4. rapikan spasi; bagian tanpa isi (`""`, `-`, `none`, `tidak ada`, `lihat kesan`, `see impression`)
   dianggap tidak ada.

**`clean_target`** dipakai **hanya untuk target training** (referensi test tidak disentuh agar
metrik tetap sebanding):

- buang kalimat logistik laporan yang tidak bisa dilihat dari gambar: nomor telepon konsultasi,
  pernyataan atestasi (*"saya telah meninjau ..."*), kalimat *"dikomunikasikan / didiskusikan dengan /
  melalui telepon ..."*
- blanko anonimisasi MIMIC `___` → `sebelumnya`
- buang sampel yang terjemahannya *looping* (frasa 1–3 kata berulang ≥5 kali)
- buang nomor daftar yang yatim, tanda kutip; target < 2 kata → sampel dibuang

### 2.3 Pembentukan sampel (`rrg/data.py`)

```
manifest.parquet (1 baris = 1 gambar)
   └─ build_samples: untuk tiap gambar × tiap tugas yang referensinya ada
        → {"image_path", "prompt", "target"}
CheXpert train : 238.750 sampel (setelah clean_target)
MIMIC validate :   4.597 sampel  × 4 (mimic_repeat)  = 18.388
Total          : 257.138 sampel  → diacak dengan random.Random(3407)
Dev            : 256 sampel acak tetap (random.Random(0)) dari dev_internal, untuk eval_loss
```

`mimic_repeat 4` artinya setiap sampel MIMIC muncul 4 kali dalam satu epoch, supaya porsinya naik dari
±1,8% ke ±7,2% (catatan efek sampingnya di 7.4).

Bentuk satu sampel (target di bawah adalah **contoh ilustrasi buatan sendiri**, bukan laporan dari
dataset; laporan asli tidak boleh disebarkan):

```
gambar : <path gambar CXR frontal>
prompt : Anda adalah seorang ahli radiologi. Tuliskan bagian KESAN dari laporan radiologi untuk foto toraks ini.
         Proyeksi: PA.
         Jawab dalam Bahasa Indonesia.
target : tidak tampak kelainan kardiopulmoner akut. ukuran jantung normal. tidak tampak efusi pleura.
```

### 2.4 Gambar

Semua gambar di manifest adalah PNG **224×224 grayscale** hasil preprocessing dataset. Saat
dimuat (`load_image`) gambar diubah ke RGB (kanal abu-abu disalin ke 3 kanal), lalu:

- **MedGemma**: dikirim apa adanya; *processor* Gemma3 me-resize ke **896×896** (bilinear),
  skala 1/255, normalisasi mean 0,5 / std 0,5 → rentang [-1, 1].
- **CheXOne**: kode kita me-resize dulu ke **448×448** (bicubic), lalu *processor* Qwen2-VL
  menormalisasi dengan mean/std CLIP. 448 sudah kelipatan 28, jadi tidak di-resize lagi.

Kenapa ukurannya beda? Supaya **kedua model melihat jumlah token visual yang sama (256)**; lihat 3.3.

---

## 3. Arsitektur kedua model (white-box)

Kedua model punya struktur yang sama: **vision encoder → projector → decoder bahasa**. Angka di
bawah dibaca langsung dari `config.json` dan header file bobot di cache Hugging Face.

### 3.1 MedGemma-1.5-4B (Gemma 3)

```
gambar 896×896
  └─ SigLIP vision encoder (27 layer, hidden 1152, MLP 4304, patch 14×14)
       64×64 = 4.096 patch embedding (1152-d)
  └─ multi-modal projector: average pooling 4×4 → 16×16 = 256 token, RMSNorm, Linear 1152 → 2560
  └─ 256 token visual (2560-d) disisipkan di posisi <start_of_image> pada prompt
  └─ decoder Gemma 3: 34 layer, hidden 2560
       attention: 8 query head, 4 key/value head (GQA), head_dim 256,
                  campuran layer lokal (sliding window 1024) dan global, RoPE
       MLP: GeGLU, intermediate 10240
       vocab 262.208 (embedding input = lm_head, tied)
```

| Komponen | Parameter |
|---|---|
| Vision encoder (SigLIP) | 416.866.032 |
| Projector | 2.950.272 |
| Embedding token (= lm_head) | 671.252.480 |
| Layer decoder (34) | 3.209.010.688 |
| **Total** | **4.300.079.472** |

### 3.2 CheXOne (Qwen2.5-VL-3B + post-training CXR)

```
gambar 448×448 (diperlakukan sebagai 2 "frame" identik, temporal patch 2)
  └─ ViT Qwen2.5-VL (32 blok, hidden 1280, MLP 3420, patch 14×14)
       32×32 = 1.024 patch; window attention (jendela 112 px), full attention di blok 7, 15, 23, 31
  └─ merger: RMSNorm, gabung 2×2 patch tetangga (4 × 1280 = 5120-d) → 256 token, MLP 5120 → 5120 → 2048
  └─ 256 token visual (2048-d) menggantikan <|image_pad|> di prompt
  └─ decoder Qwen2.5: 36 layer, hidden 2048
       attention: 16 query head, 2 key/value head (GQA), head_dim 128, M-RoPE
       MLP: SwiGLU, intermediate 11008
       vocab 151.936
```

| Komponen | Parameter |
|---|---|
| Vision encoder (ViT) | 631.975.680 |
| Merger (projector) | 36.708.608 |
| Embedding token | 311.164.928 |
| Layer decoder (36) | 2.774.773.760 |
| lm_head (tersimpan terpisah di file) | 311.164.928 |
| **Total unik** (metadata model) | **3.754.622.976** |

CheXOne adalah Qwen2.5-VL-3B yang sudah di-*post-train* oleh Stanford pada 14,7 juta sampel CXR
berbahasa Inggris (termasuk MIMIC-CXR dan CheXpert Plus; IU-Xray tidak dipakai). Jadi sebelum
fine-tuning kita, CheXOne sudah "pintar CXR" tapi hanya dalam bahasa Inggris, sedangkan MedGemma
lebih umum (banyak modalitas medis) tapi sudah multibahasa.

### 3.3 Anggaran token yang disamakan

| | MedGemma | CheXOne |
|---|---|---|
| Ukuran input | 896×896 (oleh processor) | 448×448 (oleh kita) |
| Patch | 64×64 = 4.096 | 32×32 = 1.024 |
| Reduksi | avg-pool 4×4 | merge 2×2 |
| **Token visual per gambar** | **256** | **256** |
| Token per kata Indonesia (tokenizer) | 1,66 | 2,25 |
| Panjang sekuens training (p50 / p99 / maks) | 376 / 503 / 591 | 414 / 590 / 709 |
| Batas `max_length` training | 1.024 (tidak ada sampel yang terpotong) | 1.024 (sama) |
| `max_new_tokens` inferensi | 256 | 352 |

Jumlah token visual disamakan (256) agar perbandingan adil. Budget generasi dibedakan karena
tokenizer Qwen memecah kata Indonesia lebih halus. 256 × 2,25 / 1,66 ≈ 352, jadi keduanya bisa
menulis ±154 kata (hanya 0,4% target training yang lebih panjang dari itu).

---

## 4. Prompt dan chat template

### 4.1 Teks prompt (`rrg/prompts.py`), identik untuk kedua model

```
Anda adalah seorang ahli radiologi. Tuliskan bagian {TEMUAN|KESAN} dari laporan radiologi untuk foto toraks ini.
Proyeksi: {proyeksi}.
Jawab dalam Bahasa Indonesia.
```

Baris `Proyeksi:` diisi dari metadata view:

| view di manifest | teks |
|---|---|
| AP | `AP (portabel)` |
| PA | `PA` |
| LATERAL / LL / RL | `lateral` |
| tidak diketahui (mis. semua IU-Xray) | baris dihilangkan |

### 4.2 Bentuk akhir yang masuk ke model (output `tools/check_model.py`)

**MedGemma** (template Gemma 3):

```
<bos><start_of_turn>user
<start_of_image>Anda adalah seorang ahli radiologi. Tuliskan bagian KESAN dari laporan radiologi untuk foto toraks ini.
Proyeksi: PA.
Jawab dalam Bahasa Indonesia.<end_of_turn>
<start_of_turn>model
tidak tampak kelainan kardiopulmoner akut. ...<end_of_turn>
```

`<start_of_image>` diekspansi processor menjadi 256 token gambar.

**CheXOne** (template Qwen2.5-VL / ChatML):

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Anda adalah seorang ahli radiologi. Tuliskan bagian KESAN dari laporan radiologi untuk foto toraks ini.
Proyeksi: PA.
Jawab dalam Bahasa Indonesia.<|im_end|>
<|im_start|>assistant
tidak tampak kelainan kardiopulmoner akut. ...<|im_end|>
```

`<|image_pad|>` diekspansi menjadi 256 token gambar. Baris *system* `You are a helpful assistant.`
adalah default template Qwen (tidak kita ubah).

Saat inferensi, bagian setelah `<start_of_turn>model\n` / `<|im_start|>assistant\n` dikosongkan dan
model melanjutkan sendiri sampai menghasilkan token akhir giliran (`<end_of_turn>` / `<|im_end|>`).

---

## 5. Loss dan prompt masking

Loss = **cross-entropy next-token** rata-rata, tetapi **hanya pada token laporan**:

```
L = − (1/|R|) Σ_{t ∈ R} log P(x_t | gambar, x_<t)        R = posisi token target (laporan + token akhir giliran)
```

Token system, user, gambar, dan prompt diberi label `-100` sehingga tidak dihitung. Implementasinya
adalah `UnslothVisionDataCollator(train_on_responses_only=True)` dengan penanda:

| | `instruction_part` (awal giliran user) | `response_part` (awal giliran model) |
|---|---|---|
| MedGemma | `<start_of_turn>user\n` → id `[105, 2364, 107]` | `<start_of_turn>model\n` → id `[105, 4368, 107]` |
| CheXOne | `<|im_start|>user\n` → id `[151644, 872, 198]` | `<|im_start|>assistant\n` → id `[151644, 77091, 198]` |

Semua token sebelum `response_part` di-mask. `train.py` memeriksa ini sebelum training mulai
(`check_loss_mask`). Training langsung berhenti kalau tidak ada token yang dihitung, atau kalau lebih
dari 80% token ikut dihitung (artinya prompt tidak ter-mask). Contoh dari log CheXOne: loss hanya
mencakup **78 dari 400 token** sampel pertama, persis teks laporannya.

**Kenapa di-mask?** Tanpa masking, model juga dilatih "menebak" prompt yang selalu sama dan token
gambar. Itu membuang kapasitas dan membuat loss terlihat lebih kecil tanpa arti.

Tidak ada *sequence packing*: setiap baris batch adalah satu sampel, di-*pad* ke yang terpanjang
dalam batch.

---

## 6. QLoRA: bagian mana yang dilatih, bagian mana yang dibekukan

### 6.1 Kuantisasi 4-bit (bagian "Q")

Saat model dimuat (`FastVisionModel.from_pretrained(..., load_in_4bit=True)`), bobot `Linear` di
**decoder bahasa** dikuantisasi *on the fly* oleh bitsandbytes ke **4-bit NormalFloat (NF4)**
dengan *double quantization*. Komputasinya tetap di bfloat16: bobot di-*dequantize* per layer saat
forward pass. Tidak ada file model pra-kuantisasi; yang diunduh adalah bobot bf16 asli.

Modul yang **tetap 16-bit** (dikecualikan Unsloth dari kuantisasi):

| | tetap 16-bit | 4-bit |
|---|---|---|
| MedGemma | vision tower (SigLIP), multi-modal projector, lm_head/embedding | semua Linear decoder |
| CheXOne | merger (projector), lm_head/embedding | semua Linear decoder **dan** blok ViT |

Efeknya: memori bobot decoder turun ±4×, sehingga model 4B bisa dilatih di GPU 16 GB.
Kuantisasi ini berlaku untuk bobot yang **dibekukan**. Adapter LoRA tetap presisi penuh.

### 6.2 LoRA (bagian "LoRA")

Untuk setiap matriks bobot target `W₀` (beku, 4-bit), ditambahkan dua matriks kecil yang dilatih:

```
h = W₀·x + (α/r) · B·(A·x)          A ∈ R^{r×d_in},  B ∈ R^{d_out×r}
r = 32, α = 32  →  faktor skala α/r = 1,0
inisialisasi: A acak (Kaiming-uniform), B = 0  →  di step 0 model PERSIS sama dengan base model
dropout LoRA = 0, bias tidak dilatih, tanpa DoRA / rsLoRA
```

**Target**: 7 proyeksi di **setiap layer decoder bahasa**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
(attention) dan `gate_proj`, `up_proj`, `down_proj` (MLP). Karena `--freeze_vision`, **tidak ada
LoRA di vision encoder maupun projector**.

Ukuran matriks LoRA per layer (dibaca dari `adapter_model.safetensors`):

| Proyeksi | MedGemma A / B | CheXOne A / B |
|---|---|---|
| q_proj | 32×2560 / 2048×32 | 32×2048 / 2048×32 |
| k_proj | 32×2560 / 1024×32 | 32×2048 / 256×32 |
| v_proj | 32×2560 / 1024×32 | 32×2048 / 256×32 |
| o_proj | 32×2048 / 2560×32 | 32×2048 / 2048×32 |
| gate_proj | 32×2560 / 10240×32 | 32×2048 / 11008×32 |
| up_proj | 32×2560 / 10240×32 | 32×2048 / 11008×32 |
| down_proj | 32×10240 / 2560×32 | 32×11008 / 2048×32 |
| **per layer** | **1.753.088** | **1.662.976** |
| × jumlah layer | × 34 = **59.604.992** | × 36 = **59.867.136** |
| % dari total model | 1,39% | ±1,6% |
| jumlah tensor di adapter | 476 | 504 |

Adapter disimpan dalam fp32 (±238–240 MB). Saat dipakai, adapter digabung dengan base model yang
sama persis (`google/medgemma-1.5-4b-it` atau `StanfordAIMI/CheXOne`).

### 6.3 Ringkasan: yang berubah vs yang tidak

```
TIDAK BERUBAH (beku)                          BERUBAH (dilatih, ±60 juta parameter)
- vision encoder → model "melihat" sama       - 7 × 34 (MedGemma) / 7 × 36 (CheXOne) pasang matriks LoRA
- projector → fitur visual dipetakan sama        di decoder bahasa → cara model MENULIS: bahasa,
- embedding token, lm_head                       istilah, struktur laporan, kapan menyebut temuan
- bobot asli decoder (4-bit)
```

Artinya pengetahuan visual base model dipakai apa adanya, dan fine-tuning mengajari decoder untuk
(a) menulis dalam bahasa Indonesia, (b) mengikuti gaya laporan CheXpert/MIMIC, dan (c) memetakan
fitur visual yang sudah ada ke kalimat temuan yang tepat.

---

## 7. Optimisasi dan jalannya training

### 7.1 Konfigurasi (identik untuk kedua model; `run_args.json`)

| Setting | Nilai |
|---|---|
| Optimizer | AdamW 8-bit (bitsandbytes), β=(0,9; 0,999), ε=1e-8 |
| Learning rate puncak | 2×10⁻⁴ |
| Jadwal LR | warm-up linear 3% (±482 step) lalu cosine turun ke 0 |
| Weight decay | 0,01 |
| Gradient clipping | max norm 1,0 (default) |
| Batch | 2 per step × 8 gradient accumulation = **16 sampel per update** |
| Epoch | 1 (16.072 update) |
| Panjang sekuens maks | 1.024 token (termasuk 256 token gambar) |
| Presisi | bfloat16 + bobot 4-bit |
| Gradient checkpointing | mode `"unsloth"` (aktivasi disimpan hemat / di-offload) |
| Seed | 3407 (pengacakan data, inisialisasi LoRA, sampler) |
| Evaluasi | tiap 500 step, eval_loss pada 256 sampel dev tetap, batch 1 |
| Checkpoint | tiap 500 step, simpan 2 terakhir + yang terbaik; di akhir muat yang terbaik (eval_loss terendah) |
| Hardware | 1× NVIDIA RTX 5060 Ti 16 GB, Windows |
| Software | torch 2.11 (cu128), transformers 5.5.0, unsloth 2026.9.11, trl 0.24.0, peft 0.21.2, bitsandbytes 0.50.2 |

### 7.2 Satu langkah training

```
1. ambil 2 sampel → load gambar → render chat template → tokenisasi + 256 token gambar per sampel
2. forward: vision encoder (beku) → projector (beku) → decoder (4-bit + LoRA) → logits
3. loss cross-entropy hanya pada token laporan
4. backward: gradien HANYA mengalir ke matriks LoRA A dan B
5. ulangi 8 kali (gradient accumulation) → clip norm 1,0 → update AdamW 8-bit → LR scheduler maju 1 step
```

### 7.3 Kurva eval loss (dari `trainer_state.json`)

| Step | MedGemma eval_loss | CheXOne eval_loss |
|---|---|---|
| 500 | 1,6524 | 1,2964 |
| 2.500 | 1,4176 | 1,1035 |
| 4.500 | 1,3510 | 1,0500 |
| 6.500 | 1,3118 | 1,0121 |
| 8.500 | 1,2764 | 0,9852 |
| 10.500 | 1,2440 | 0,9596 |
| 12.500 | 1,2202 | 0,9453 |
| 14.500 | 1,2116 | 0,9388 |
| **16.072 (akhir = terbaik)** | **1,2098** | **0,9370** |
| train loss di akhir | 1,124 | 0,884 |

Kedua kurva turun monoton dan masih sedikit turun di akhir, jadi **tidak ada overfitting pada data
CheXpert** (dev set hanya berisi CheXpert; untuk MIMIC lihat 7.4). **Catatan penting:** angka eval_loss
MedGemma vs CheXOne **tidak bisa dibandingkan** langsung karena tokenizer-nya berbeda (loss per token,
dan jumlah token per kata berbeda). Model dibandingkan lewat metrik test (Bab 10).

### 7.4 Catatan: efek `mimic_repeat 4`

Karena subset MIMIC kecil dan dilihat 4 kali, model menghafal sebagian laporan training MIMIC:
±17% (MedGemma) dan ±10% (CheXOne) findings di test MIMIC sama persis dengan satu laporan training
(run tanpa repeat: 0%). Ini overfitting **sebagian**: mayoritas output tetap disusun sendiri dan hasil
utama tetap sah karena test set tidak pernah dilihat. Untuk run berikutnya: tanpa repeat
(`mimic_repeat 1`) atau terjemahkan lebih banyak data MIMIC. Cek dengan `tools/check_memorization.py`.

---

## 8. Inferensi dan post-processing

### 8.1 Generasi (`infer.py`)

```
1. muat base model 4-bit + adapter (Unsloth), mode inferensi, padding kiri
2. untuk setiap gambar test (termasuk lateral), batch 8:
     prompt TEMUAN → generate → teks findings
     prompt KESAN  → generate → teks impression
3. dekode token baru saja (tanpa prompt), buang token spesial
```

| Setting | Nilai |
|---|---|
| Decoding | greedy (`do_sample=False`, `num_beams=1`), jadi deterministik |
| Repetition penalty | 1,05 |
| `max_new_tokens` | 256 (MedGemma) / 352 (CheXOne) |
| Berhenti | saat token akhir giliran (`<end_of_turn>` / `<|im_end|>`) atau budget habis |

### 8.2 Post-processing (`rrg/postprocess.py`), hanya pada output model

Greedy decoding kadang mengulang kalimat. Tiga aturan berbasis regex:

1. frasa (1–8 kata) yang berulang berturut-turut ≥3 kali → disisakan satu
2. kalimat / item bernomor yang sama dengan sebelumnya (abaikan huruf besar, tanda baca, nomor) →
   dibuang; daftar bernomor diberi nomor ulang
3. kalimat terakhir yang terpotong (tidak berakhir `.`, `!`, `?`; biasanya karena budget token
   habis) → dibuang

Kalimat yang mirip tapi isinya beda (*paru kanan* vs *paru kiri*) **tidak** dibuang. Output asli
tetap disimpan di kolom `generated_<task>_raw`. Efek post-processing pada metrik maksimal 0,005.

Yang **tidak** bisa diperbaiki post-processing: model mengeluarkan kalimat template yang sama
untuk banyak gambar berbeda (mis. 478/3.511 findings MIMIC dari MedGemma berisi satu kalimat yang
sama, yang menyatakan tidak ada perubahan dibanding studi sebelumnya). Ini perilaku model,
bukan pengulangan di dalam satu laporan.

---

## 9. Perbandingan MedGemma vs CheXOne (side by side)

### 9.1 Yang SAMA (dikontrol supaya adil)

- manifest, split, sampel, urutan sampel (seed 3407), `clean_target`, `mimic_repeat 4`
- teks prompt, satu adapter untuk dua tugas, prompt masking
- 256 token visual per gambar, vision encoder beku
- LoRA r=32, α=32, target 7 proyeksi di semua layer decoder
- optimizer, LR, jadwal, batch efektif, 1 epoch = 16.072 step, max_length 1.024
- decoding greedy, repetition penalty 1,05, budget ±154 kata, post-processing, evaluasi

### 9.2 Yang BERBEDA (melekat pada base model)

| Aspek | MedGemma-1.5-4B | CheXOne |
|---|---|---|
| Keluarga | Gemma 3 4B | Qwen2.5-VL 3B |
| Pengetahuan awal | medis umum (radiologi, dermatologi, patologi, teks medis), multibahasa | khusus CXR, 14,7 juta sampel, bahasa Inggris |
| Pernah melihat data test? | pre-training dilaporkan memuat MIMIC-CXR | post-training memuat MIMIC-CXR & CheXpert Plus; **IU-Xray tidak** |
| Vision encoder | SigLIP, 27 layer, 417 jt param, input 896 | ViT 32 blok window-attention, 632 jt param, input 448 |
| Projector | avg-pool 4×4 + Linear (2,9 jt param) | merge 2×2 + MLP (36,7 jt param) |
| Decoder | 34 layer, hidden 2560, GQA 8/4, GeGLU | 36 layer, hidden 2048, GQA 16/2, SwiGLU |
| Vocab / token per kata ID | 262 rb / 1,66 | 152 rb / 2,25 |
| System prompt | tidak ada | `You are a helpful assistant.` (default) |
| Bagian 16-bit saat QLoRA | vision tower, projector, lm_head | merger, lm_head (blok ViT ikut 4-bit) |
| LoRA trainable | 59,60 jt | 59,87 jt |
| Kecepatan training | 18,8 s/step (±84 jam) | 8,0 s/step (±36 jam), 2,3× lebih cepat |
| Lisensi | Health AI Developer Foundations terms | CC BY-NC 4.0 (non-komersial) |

### 9.3 Apa artinya untuk hasil

- CheXOne unggul di **MIMIC-CXR** dan impression **CheXpert Plus**. Keduanya ada di data
  post-training-nya, jadi model sudah familiar dengan gaya dan distribusinya.
- MedGemma unggul signifikan di **IU-Xray** (rumah sakit berbeda, tidak pernah dilihat CheXOne) di
  semua metrik klinis. Ini perbandingan yang paling adil.
- CheXOne membawa kebiasaan gaya MIMIC ke IU-Xray: kalimat pembuka prosedur, perbandingan dengan
  "studi sebelumnya" yang tidak ada, dan laporan lebih panjang (55 kata vs referensi 32).

---

## 10. Evaluasi (ringkas)

Rincian lengkap ada di `rexrank_eval/README.md`; semua angka ada di `results/RESULTS.md`.

1. **Metrik bahasa Indonesia** (`evaluate.py`, `rrg/metrics.py`) vs referensi Indonesia: BLEU-1–4
   corpus (sacrebleu), BLEU-2 versi ReXrank per laporan, ROUGE-L (tanpa stemming), METEOR (hanya
   pelengkap: sinonimnya bahasa Inggris), BERTScore multilingual BERT. Dihitung per tugas:
   findings, impression, findings+impression.
2. **Metrik klinis ReXrank lewat terjemahan** (`rexrank_eval/`): output Indonesia diterjemahkan ke
   Inggris dengan **MedGemma-1.5-4B base** (greedy, prompt di bawah), lalu dibandingkan dengan
   **laporan Inggris asli**: 1/RadCliQ-v1, BLEU-2, BERTScore, SembScore, RadGraph, RaTEScore, GREEN,
   CRIMSON. Level studi (1 gambar per studi, frontal diutamakan); GREEN/CRIMSON pada 500 studi acak
   tetap per test set per track.

   ```
   Translate this Indonesian chest X-ray report text into English. Use standard English radiology
   terminology. Translate faithfully: keep every finding, negation, laterality, size and list numbering;
   do not add, remove or explain anything. Output only the English translation.

   {teks}
   ```
3. **Ceiling terjemahan**: referensi Indonesia diterjemahkan balik lewat jalur yang sama → batas atas
   skor yang bisa dicapai kalau output model persis sama dengan referensi.
4. **Uji signifikansi**: paired bootstrap 2.000 kali pada studi yang sama; signifikan kalau CI 95%
   tidak memuat 0 (`tools/compare_runs.py`, `rexrank_eval/compare_runs.py`, hasil di
   `results/significance.json`).

---

## 11. Alasan di balik pilihan desain

| Pilihan | Alasan / bukti |
|---|---|
| **QLoRA 4-bit**, bukan full fine-tuning | full fine-tuning model 4B tidak muat di GPU 16 GB; QLoRA dengan Unsloth juga sesuai arahan pembimbing |
| **Vision encoder beku** (`--freeze_vision`) | eksperimen awal: LoRA di vision memberi eval_loss hampir sama tapi butuh VRAM lebih besar |
| **LoRA r=32, α=32** | run awal 40k sampel memakai r=16; untuk data penuh (6× lebih banyak) kapasitas adapter dinaikkan |
| **Satu adapter untuk 2 tugas** | TEMUAN dan KESAN berbagi pengetahuan visual yang sama; prompt cukup untuk membedakan |
| **`mimic_repeat 4`** | MIMIC hanya ±2% data; tanpa repeat, KESAN MIMIC ikut gaya daftar bernomor ala CheXpert. Efek sampingnya sebagian laporan MIMIC terhafal (7.4) |
| **`clean_target`** | referensi berisi sampah (nomor telepon, atestasi, `___` MIMIC, terjemahan looping) yang tidak bisa dipelajari dari gambar |
| **Batch 2 × accum 8, eval batch 1** | run sebelumnya dengan batch 4 kehabisan VRAM saat evaluasi di step 1000 (5,5 jam hilang) |
| **1 epoch** | ±84 jam untuk MedGemma; eval_loss masih turun pelan tapi sudah mendatar |
| **256 token visual untuk keduanya** | supaya selisih hasil berasal dari base model, bukan dari resolusi input |
| **max_new_tokens 256 vs 352** | menyamakan budget kata (±154 kata), bukan budget token |
| **Greedy + repetition penalty 1,05** | deterministik dan reproducible; penalti kecil mengurangi loop tanpa merusak istilah yang memang berulang |

---

## 12. Yang tidak dilakukan / keterbatasan

- `mimic_repeat 4` membuat sebagian laporan training MIMIC terhafal (7.4).
- Tidak ada LoRA di vision encoder; model tidak belajar "melihat" dengan cara baru.
- Gambar hanya 224×224 (di-upscale processor); detail halus (nodul kecil, garis kateter) bisa hilang.
- Hanya 1 run per model, satu seed; variasi antar-seed tidak diukur.
- Data MIMIC untuk training hanya split *validate* (yang sudah diterjemahkan).
- Terjemahan dataset (TranslateGemma-4B) belum divalidasi radiolog; evaluasi klinis juga lewat
  terjemahan (MedGemma), dan keduanya dari keluarga Gemma 3.
- Baseline zero-shot hanya ada untuk CheXpert (MedGemma); baseline "model Inggris → terjemahan
  mesin" belum dibuat.
- Classifier hints (`--hints`) tersedia di kode tetapi tidak dipakai.
- Prior hallucination: ±59% KESAN CheXpert di data training menyebut perbandingan dengan studi
  sebelumnya, sehingga model belajar menyebut "sebelumnya" walaupun tidak ada gambar sebelumnya.

---

## 13. Peta kode dan cara reproduksi

| Tahap | File |
|---|---|
| Path, peran split | `rrg/config.py` |
| Manifest (gabung 3 dataset, split, `clean_text`) | `prepare_data.py`, `rrg/text.py` |
| Registry base model (template, ukuran gambar, budget token) | `rrg/models.py` |
| Prompt | `rrg/prompts.py` |
| Sampel training, load gambar, dataset | `rrg/data.py` |
| Cek template / masking / token gambar (CPU) | `tools/check_model.py` |
| Training QLoRA | `train.py` |
| Inferensi | `infer.py` |
| Post-processing | `rrg/postprocess.py` |
| Metrik Indonesia | `evaluate.py`, `rrg/metrics.py` |
| Terjemahan + ReXrank | `rexrank_eval/translate.py`, `rexrank_eval/rexrank.py` |
| Uji signifikansi | `tools/compare_runs.py`, `rexrank_eval/compare_runs.py` |
| Hasil training | `checkpoints/<run>/final_adapter/` (`adapter_model.safetensors`, `adapter_config.json`, `run_args.json`) |

Urutan menjalankan (dari folder `rrg_indo/`):

```powershell
uv run python prepare_data.py
uv run python tools/check_model.py --model medgemma          # dan --model chexone (CPU)

# training (perintah persis kedua run; argumen lengkap juga tersimpan di
#   checkpoints/<run>/final_adapter/run_args.json)
uv run python train.py --run_name medgemma_full --freeze_vision --mimic_repeat 4 --lora_r 32 --lora_alpha 32
uv run python train.py --model chexone --run_name chexone_full --freeze_vision --mimic_repeat 4 --lora_r 32 --lora_alpha 32

# inferensi + metrik Indonesia + terjemahan + ReXrank (Git Bash)
bash scripts/eval_run.sh medgemma_full
bash scripts/eval_run.sh chexone_full
```

`--dry_run` di `train.py` menampilkan jumlah sampel dan step tanpa GPU (mis. untuk memastikan
257.138 sampel / 16.072 step).

---

## 14. Catatan: pendekatan ketiga (`rrg_qwen`)

Selain dua run di atas, ada pendekatan **modular** ala RaDialog (proyek terpisah `rrg_qwen`, tidak
termasuk di repositori ini): BioViL-T (beku) → modul alignment 32 *soft token* → classifier 14 label →
Qwen3-1.7B-Base + LoRA yang menerima soft token dan label hasil classifier sebagai konteks. Kodenya
sudah lengkap dan lulus uji CPU, tetapi **belum pernah dilatih di GPU** dan tidak masuk paper.
