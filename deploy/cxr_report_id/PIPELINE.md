# Bagaimana pipeline ini dibangun

Dokumen ini menjelaskan dari mana kedua model berasal: data, cara training, cara inference, dan
hasil evaluasinya. Kode lengkap untuk training dan evaluasi ada di repositori riset (`rrg_indo/`).
Folder deploy ini hanya berisi bagian yang dibutuhkan untuk inference.

## 1. Gambaran umum

```
foto toraks ──► resize 224×224 grayscale ──► ukuran input model:
                                              MedGemma 896×896 / CheXOne 448×448 (256 token gambar)
                                                        │
prompt bahasa Indonesia (TEMUAN atau KESAN + proyeksi) ─┤
                                                        ▼
                     base model: vision encoder (beku) ──► projector ──► language model + LoRA
                                                        │
                                                        ▼
                              teks laporan ──► post-processing ──► TEMUAN / KESAN
```

- **Dua base model, satu resep.** MedGemma dan CheXOne dilatih dengan sampel, urutan, prompt,
  pengaturan LoRA, dan optimizer yang sama persis. Perbedaan hasil berasal dari base model-nya.
- **Satu model untuk dua tugas.** Adapter yang sama menulis TEMUAN atau KESAN, tergantung prompt-nya.
- **End-to-end.** Gambar langsung masuk ke model vision-language. Tidak ada classifier terpisah dan
  tidak ada tahap alignment terpisah, karena kedua base model sudah dilatih dengan foto toraks.

## 2. Base model

| | MedGemma 1.5 4B-it | CheXOne |
|---|---|---|
| Hugging Face | `google/medgemma-1.5-4b-it` | `StanfordAIMI/CheXOne` |
| Pembuat | Google | Stanford AIMI |
| Arsitektur | Gemma 3 4B + vision encoder SigLIP | Qwen2.5-VL-3B |
| Pengetahuan medis | model medis umum (berbagai jenis citra medis, termasuk foto toraks) | spesialis foto toraks: post-training dengan 14,7 juta sampel (MIMIC-CXR, CheXpert Plus, dan lainnya; IU X-ray tidak dipakai) |
| Bahasa | multibahasa, termasuk Bahasa Indonesia | bahasa Inggris |
| Gambar | 896×896 → 256 token gambar | 448×448 → 256 token gambar |
| Lisensi | HAI-DEF Terms of Use | CC BY-NC 4.0 (non-komersial) |

Pertanyaan riset di balik perbandingan ini: mana yang lebih membantu untuk laporan berbahasa
Indonesia, pengetahuan foto toraks yang sangat spesifik tapi berbahasa Inggris (CheXOne), atau
model medis umum yang multibahasa (MedGemma)?

## 3. Data

Data training dan pengujian berupa laporan radiologi berbahasa Inggris yang sudah diterjemahkan ke
Bahasa Indonesia. Gambarnya berupa PNG 224×224 grayscale yang memperlihatkan seluruh foto.

| Split | Sumber | Gambar | Studi | Dipakai untuk |
|---|---|---|---|---|
| train_internal | CheXpert Plus (train, 85% pasien) | 189.948 | 159.694 | training |
| train_mimic | MIMIC-CXR (split validate) | 2.852 | 1.719 | training |
| dev_internal | CheXpert Plus (train, 15% pasien lainnya) | 33.143 | 27.701 | memantau loss (256 sampel) |
| test_chexpert | CheXpert Plus (valid) | 234 | 200 | pengujian |
| test_mimic | MIMIC-CXR (test) | 4.667 | 2.921 | pengujian |
| test_openi | OpenI / IU X-ray | 7.430 | 3.826 | pengujian eksternal (sumber tidak ikut training) |

- Train, dev, dan test **dipisah per pasien**, sehingga tidak ada pasien yang muncul di lebih dari
  satu split. Hal ini sudah dicek: tidak ada pasien, studi, atau file gambar yang sama antara train
  dan test.
- **Satu sampel training = satu gambar + satu bagian laporan.** Gambar yang punya TEMUAN dan KESAN
  menghasilkan dua sampel.
- **MIMIC-CXR diulang 4×** dalam satu epoch, karena porsinya hanya ±2% dari data. Tanpa
  pengulangan ini, KESAN untuk MIMIC ikut bergaya daftar bernomor seperti CheXpert.
- **Total: 257.138 sampel training** (238.750 dari CheXpert Plus dan 4.597 × 4 dari MIMIC-CXR).

### Pembersihan teks

- **Semua teks** (target training dan referensi test):
  - ubah ke huruf kecil;
  - hapus awalan "temuan:" / "kesan:";
  - samakan istilah hasil terjemahan dengan kamus kecil (misalnya "kekeruhan" → "opasitas",
    "tabung endotrakeal" → "selang endotrakeal").
- **Target training saja:**
  - buang baris nomor telepon konsultasi, kalimat atestasi dokter, dan kalimat "dikomunikasikan
    kepada dr. X";
  - buang placeholder anonimisasi MIMIC (`___`);
  - buang hasil terjemahan yang berulang-ulang (*looping*).

  Referensi test tidak diubah, supaya angka evaluasi bisa dibandingkan dengan baseline.

## 4. Prompt

Setiap sampel memakai chat template bawaan masing-masing model. Bagian *user* berisi gambar dan
teks berikut, sama untuk kedua model:

```
Anda adalah seorang ahli radiologi. Tuliskan bagian TEMUAN dari laporan radiologi untuk foto toraks ini.
Proyeksi: PA.
Jawab dalam Bahasa Indonesia.
```

- Untuk KESAN, kata "TEMUAN" diganti "KESAN".
- Baris proyeksi bernilai `AP (portabel)`, `PA`, atau `lateral`, dan dihilangkan kalau proyeksi
  tidak diketahui.
- Jawaban model adalah teks laporan yang sudah dibersihkan.
- Teks prompt ini harus sama persis saat inference (`cxr_report/prompts.py`).

## 5. Training

| Pengaturan | MedGemma | CheXOne |
|---|---|---|
| Metode | QLoRA dengan Unsloth (base model 4-bit NF4, adapter LoRA 16-bit) | sama |
| LoRA | r = 32, alpha = 32, dropout 0, di attention dan MLP language model | sama |
| Vision encoder | beku (tidak dilatih) | sama |
| Loss | hanya pada token jawaban (prompt dan gambar di-*mask*) | sama |
| Epoch / step | 1 epoch = 16.072 step | sama |
| Batch | 2 per step × gradient accumulation 8 = 16 sampel | sama |
| Optimizer | AdamW 8-bit, learning rate 2e-4, cosine schedule, warmup 3%, weight decay 0,01 | sama |
| Presisi | bfloat16 | sama |
| Panjang maksimum | 1.024 token (gambar + prompt + jawaban) | sama |
| Seed | 3407 | sama |
| Hardware / durasi | 1× RTX 5060 Ti 16 GB, ±18,8 detik/step, ±84 jam | 1× RTX 5060 Ti 16 GB, ±8 detik/step, ±36 jam |
| Hasil | eval loss 1,2098 (step terakhir = terbaik) | eval loss 0,937 (step terakhir = terbaik) |

- Eval loss **tidak bisa dibandingkan antar model**, karena tokenizer-nya berbeda: Qwen memecah satu
  kata Indonesia menjadi ±2,25 token, Gemma ±1,66. Kalau dihitung per kata, keduanya kira-kira
  setara (selisih ±5%, masih dalam batas ketelitian perkiraan).
- Pada kedua model loss turun terus sampai akhir, dan tidak ada tanda overfitting.
- Konfigurasi lengkapnya ada di `models/<model>/run_args.json`.

## 6. Inference

1. Gambar diubah ke 224×224 grayscale, lalu ke RGB, sama seperti data training
   (`cxr_report/preprocess.py`). Untuk CheXOne, gambar lalu di-resize ke 448×448 (bicubic), juga
   seperti saat training. Processor MedGemma memperbesarnya sendiri ke 896×896.
2. Dua kali generate per gambar, satu untuk TEMUAN dan satu untuk KESAN. Keduanya memakai decoding
   greedy dan `repetition_penalty` 1,05, dengan batas token baru yang setara ±150 kata: 256 token
   (MedGemma) dan 352 token (CheXOne, karena tokenizer-nya butuh lebih banyak token per kata).
3. **Post-processing** (`cxr_report/postprocess.py`):
   - frasa yang berulang 3× atau lebih berturut-turut diringkas jadi satu;
   - kalimat atau butir daftar yang mengulang kalimat sebelumnya dibuang;
   - kalimat terakhir yang terpotong (karena batas token) dibuang.

### Kesetaraan dengan pipeline evaluasi

Saat training dan evaluasi, base model di-load dalam 4-bit oleh Unsloth. Paket deploy ini memakai
transformers + peft biasa dengan pengaturan 4-bit yang sama: NF4 dan double quantization, dengan
bagian yang tetap 16-bit sama persis seperti di Unsloth:
- MedGemma: vision encoder dan `lm_head`;
- CheXOne: *merger* (projector gambar) dan `lm_head`.

Ini sudah dicek lapisan per lapisan.

Pengecekan tanpa GPU (lolos untuk kedua model):
- token prompt dan piksel gambar yang masuk ke model identik dengan pipeline evaluasi, termasuk
  saat beberapa gambar diproses dalam satu batch;
- struktur adapter cocok dengan model transformers biasa (MedGemma 476/476, CheXOne 504/504 tensor
  LoRA).

**Hasil di GPU:**
- **Versi paket sebelumnya (MedGemma saja, 234 gambar test CheXpert Plus, 2026-10-16):**
  - hanya ±25% laporan yang sama persis kata per kata dengan pipeline evaluasi, tapi semua skor
    berbeda paling banyak ±0,01 dan BERTScore-nya identik;
  - penyebab utamanya: Unsloth mengganti beberapa perhitungan (normalisasi, attention, MLP) dengan
    kernel versinya sendiri, dan dengan decoding greedy, begitu satu kata berbeda sisa kalimatnya
    ikut berubah;
  - versi itu juga ikut mengompres vision encoder MedGemma ke 4-bit, sedangkan saat training
    vision encoder tetap 16-bit. Versi ini sudah memperbaikinya.
- **Versi ini (kedua model):** belum diuji ulang di GPU terhadap pipeline evaluasi. Tim riset bisa
  melakukannya dengan `rrg_indo/tools/check_deploy.py --model medgemma|chexone --n 234`; di mesin
  aplikasi, `selftest.py` memastikan model termuat, menulis laporan, dan mengukur VRAM serta waktu.

## 7. Hasil evaluasi

### Metrik teks bahasa Indonesia

Output yang sudah di-post-process dibandingkan dengan referensi bahasa Indonesia. Skalanya 0–1,
semakin tinggi semakin baik. "Gabungan" berarti TEMUAN dan KESAN digabung jadi satu laporan.
n adalah jumlah gambar yang punya referensi untuk bagian tersebut.

| Test set | Bagian | n | Model | BLEU-1 | BLEU-4 | ROUGE-L | METEOR | BERTScore |
|---|---|---:|---|---:|---:|---:|---:|---:|
| CheXpert Plus | TEMUAN | 74 | MedGemma | 0,324 | 0,077 | 0,230 | 0,214 | 0,763 |
| | | | CheXOne | 0,341 | 0,091 | 0,233 | 0,221 | 0,766 |
| CheXpert Plus | KESAN | 232 | MedGemma | 0,330 | 0,085 | 0,241 | 0,197 | 0,763 |
| | | | CheXOne | 0,363 | 0,097 | 0,245 | 0,214 | 0,766 |
| CheXpert Plus | gabungan | 72 | MedGemma | 0,362 | 0,097 | 0,228 | 0,218 | 0,736 |
| | | | CheXOne | 0,390 | 0,109 | 0,228 | 0,229 | 0,737 |
| MIMIC-CXR | TEMUAN | 3.511 | MedGemma | 0,302 | 0,062 | 0,201 | 0,176 | 0,744 |
| | | | CheXOne | 0,322 | 0,066 | 0,216 | 0,189 | 0,755 |
| MIMIC-CXR | KESAN | 3.722 | MedGemma | 0,241 | 0,056 | 0,219 | 0,188 | 0,748 |
| | | | CheXOne | 0,236 | 0,058 | 0,230 | 0,192 | 0,752 |
| MIMIC-CXR | gabungan | 2.566 | MedGemma | 0,363 | 0,084 | 0,220 | 0,212 | 0,744 |
| | | | CheXOne | 0,362 | 0,088 | 0,234 | 0,219 | 0,750 |
| OpenI (eksternal) | TEMUAN | 6.473 | MedGemma | 0,337 | 0,075 | 0,232 | 0,244 | 0,774 |
| | | | CheXOne | 0,276 | 0,060 | 0,221 | 0,240 | 0,766 |
| OpenI (eksternal) | KESAN | 7.415 | MedGemma | 0,305 | 0,119 | 0,350 | 0,260 | 0,804 |
| | | | CheXOne | 0,316 | 0,117 | 0,340 | 0,253 | 0,800 |
| OpenI (eksternal) | gabungan | 6.458 | MedGemma | 0,376 | 0,102 | 0,270 | 0,272 | 0,774 |
| | | | CheXOne | 0,319 | 0,083 | 0,252 | 0,269 | 0,765 |

- Angka di tabel ini berasal dari pipeline evaluasi (Unsloth).
- Sebagai pembanding, MedGemma tanpa fine-tuning (*zero-shot*) di CheXpert Plus (gabungan) hanya
  mencapai BLEU-4 0,005 dan ROUGE-L 0,110.
- Di CheXpert Plus, keunggulan CheXOne sebagian besar karena laporannya lebih panjang (lebih dekat ke
  panjang referensi), bukan karena kata-katanya lebih tepat.

### Metrik klinis gaya ReXrank (bahasa Inggris)

Metrik di leaderboard [ReXrank](https://rexrank.ai) hanya bisa membaca bahasa Inggris. Karena itu
output bahasa Indonesia diterjemahkan dulu ke bahasa Inggris (penerjemahnya MedGemma 1.5 4B),
lalu dinilai terhadap laporan asli berbahasa Inggris.

**Batas atas** didapat dengan cara yang sama, tapi memakai referensi bahasa Indonesia yang
diterjemahkan balik. Angka ini adalah skor laporan yang isinya sempurna setelah melewati
terjemahan. Selisih antara batas atas dan 1,0 adalah kerugian akibat terjemahan, bukan akibat
model.

Hasil untuk TEMUAN, tingkat studi (satu gambar per studi). ✔ menandai model yang unggul secara
signifikan (bootstrap berpasangan, 95%); tanpa tanda berarti selisihnya tidak signifikan.

| Test set | Model | RadGraph | RaTEScore | SembScore | BLEU-2 | BERTScore | GREEN | CRIMSON | 1/RadCliQ-v1 | Peringkat ReXrank* |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CheXpert Plus (62 studi) | MedGemma | 0,236 | 0,541 | 0,391 | 0,141 | 0,242 | 0,272 | 0,180 | 0,732 | 11 dari 30 |
| | CheXOne | 0,228 | 0,545 | 0,399 | 0,153 | 0,253 | 0,305 | 0,219 | 0,740 | 10 dari 30 |
| | *batas atas* | *0,654* | *0,825* | *0,914* | *0,569* | *0,696* | | | | |
| MIMIC-CXR (2.141) | MedGemma | 0,134 | 0,522 | 0,290 | 0,113 | 0,149 | 0,230 | -0,008 | 0,572 | 27 dari 30 |
| | CheXOne | 0,165 ✔ | 0,520 | 0,357 ✔ | 0,131 ✔ | 0,188 ✔ | 0,257 ✔ | 0,028 | 0,633 ✔ | 25 dari 30 |
| | *batas atas* | *0,663* | *0,835* | *0,914* | *0,554* | *0,690* | | | | |
| OpenI / IU X-ray (3.337) | MedGemma | 0,214 ✔ | 0,595 ✔ | 0,497 ✔ | 0,147 ✔ | 0,228 ✔ | 0,594 ✔ | 0,298 ✔ | 0,761 ✔ | 28 dari 30 |
| | CheXOne | 0,197 | 0,587 | 0,464 | 0,141 | 0,220 | 0,540 | 0,205 | 0,725 | 28 dari 30 |
| | *batas atas* | *0,677* | *0,879* | *0,935* | *0,602* | *0,704* | | | | |

GREEN dan CRIMSON (penilai berbasis LLM) dihitung pada 500 studi acak per test set (CheXpert: semua
62 studi), studi yang sama untuk kedua model.

\* Posisi 1/RadCliQ-v1 di antara 29 model pada leaderboard bagian Findings (diambil 2026-10-16).
Ini hanya gambaran kasar, bukan peringkat resmi, karena:
- output kita melewati terjemahan;
- test set-nya tidak persis sama dengan yang dipakai leaderboard.

Cara membacanya:
- **Pemenangnya tergantung asal data.**
  - CheXOne unggul di MIMIC-CXR (hampir semua metrik signifikan) dan di KESAN CheXpert Plus
    (1/RadCliQ 0,787 vs 0,739). Keduanya sumber data yang sudah dipelajari CheXOne saat pretraining.
  - MedGemma unggul di OpenI (semua metrik signifikan). OpenI adalah satu-satunya test set yang
    tidak pernah dilihat CheXOne maupun fine-tuning kita.
  - Untuk rumah sakit baru, termasuk rumah sakit di Indonesia, hasil OpenI paling relevan. Karena
    itu MedGemma menjadi model default.
- **Di OpenI, CheXOne terbawa gaya MIMIC-CXR:**
  - TEMUAN-nya rata-rata 55 kata, padahal referensinya 32 kata (MedGemma 43);
  - 43% TEMUAN-nya diawali kalimat prosedur khas MIMIC ("foto PA dan proyeksi dada lateral diperoleh
    dengan pasien ..."), sedangkan MedGemma 20%.
- **Kelemahan MedGemma di MIMIC sebagian karena kalimat template:** 478 TEMUAN berbunyi
  "dibandingkan dengan studi sebelumnya, tidak ada perubahan yang signifikan ...".
- **Metrik kata (BLEU-2, BERTScore) paling banyak dirugikan oleh terjemahan.** Laporan yang
  sempurna pun hanya mencapai BLEU-2 ±0,57 dan BERTScore ±0,70. **Metrik isi klinis (SembScore,
  RaTEScore) paling sedikit dirugikan,** jadi paling layak untuk membandingkan.
- Batas atas 1/RadCliQ-v1 tidak dicantumkan, karena rumus RadCliQ menghasilkan angka negatif
  untuk skor setinggi itu.

## 8. Keterbatasan yang diketahui

- **Menyebut foto sebelumnya yang tidak ada.** Sekitar 59% KESAN CheXpert di data training merujuk
  ke foto terdahulu ("stabil", "persisten", "dibandingkan dengan studi sebelumnya"). Kedua model
  menirunya, padahal hanya melihat satu gambar. Kadang tanggal pemeriksaan juga dikarang.
- **Kalimat template.** Contohnya, 478 dari 3.511 TEMUAN MIMIC dari MedGemma (14%) berbunyi sama persis. Di OpenI, banyak
  gambar normal mendapat TEMUAN yang identik dari kedua model. Post-processing tidak bisa
  memperbaiki ini, karena memang perilaku model.
- **Gaya laporan.** Gaya laporan didominasi CheXpert (±93% sampel training), misalnya KESAN berupa
  daftar bernomor yang panjang. CheXOne juga membawa gaya MIMIC-CXR dari pretraining-nya.
- **Resolusi 224×224.** Gambar asli rontgen dada biasanya 2.000–3.000 piksel. Temuan yang sangat
  kecil (nodul kecil, garis pneumotoraks yang tipis) bisa hilang.
- **Belum diuji di Indonesia.** Model belum diuji pada foto dan laporan dari rumah sakit Indonesia,
  dan belum divalidasi secara klinis.
- **Kemungkinan tumpang-tindih dengan data pretraining.**
  - Data pretraining CheXOne mencakup MIMIC-CXR dan CheXpert Plus, dan data pretraining MedGemma
    dilaporkan mencakup MIMIC-CXR.
  - Akibatnya angka test MIMIC dan CheXpert mungkin terlalu optimistis, terutama untuk CheXOne.
  - OpenI adalah pengujian yang paling bersih.
