# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** Nguyễn Tiến Thắng 2A202600220
**Cohort:** A20-K1
**Tier đã chạy:** T4
**Date:** 2026-05-09

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | Free Colab T4 (14.56 GB VRAM) |
| CUDA / driver | CUDA 12.8 · Triton 3.6.0 · Torch 2.10.0+cu128 |
| Base model | `unsloth/Qwen2.5-3B-bnb-4bit` |
| SFT dataset slice | `bkai-foundation-models/vi-alpaca` · 1 000 samples · 1 epoch |
| Preference dataset slice | `argilla/ultrafeedback-binarized-preferences-cleaned` · 2 000 pairs · 1 epoch |
| `COMPUTE_TIER` env | T4 |
| Total cost | $0 (Free Colab) |

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time | ~10 min (125 steps) | ~52 min (250 steps) |
| VRAM peak | ~10 GB | ~13.8 GB |
| Final loss | — (SFT loss curves saved) | 0.48 (DPO final) |
| Reward gap (chosen − rejected, end of training) | n/a | **+0.089** |
| AlpacaEval-lite win-rate (vs SFT-only) | 0.50 (baseline) | **0.55 (+4.5%)** |

**Tulu 3 reference numbers** (from deck §7.2b, for context only):
- +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO baseline on Llama-3-8B-Instruct)
- 70B-class scale; do not expect to replicate at 3B / 7B.

---

## 3. Reward curves analysis (≥ 100 words)

> **Xem `03_dpo_reward_curves.png` trong `submission/screenshots/`.**

Trong quá trình DPO training trên 250 steps, reward curves cho thấy một pattern kinh điển của DPO thành công: **chosen reward tăng nhẹ** trong khi **rejected reward giảm**, dẫn đến reward gap dương (+0.089) ở cuối training. Cụ thể, `chosen_reward` kết thúc ở -0.478 và `rejected_reward` ở -0.567.

Điều đáng chú ý là cả hai giá trị đều ở vùng âm — đây là hiện tượng **likelihood displacement** được đề cập trong bài giảng §3.4. Mô hình không chỉ học "thích" response tốt hơn mà còn có xu hướng giảm xác suất cho tất cả các chuỗi, đặc biệt là rejected. Trong 100 step đầu, reward gap hầu như phẳng (flat), chứng tỏ mô hình cần thời gian để "tìm phương hướng" từ tín hiệu preference. Sau đó, gap dần mở rộng theo hướng có lợi, xác nhận rằng DPO đã học được signal phân biệt chosen khỏi rejected. Mặc dù reward gap +0.089 tương đối nhỏ (có thể do dataset tiếng Anh trong khi model được SFT trên dữ liệu tiếng Việt), nhưng sự cải thiện win-rate trên AlpacaEval-lite (+4.5%) xác nhận rằng DPO đã có tác dụng thực tế.

---

## 4. Qualitative comparison (≥ 8 examples)

> **Xem `04_side_by_side_table.png` trong `submission/screenshots/`.**

| # | Prompt category | Prompt (truncated) | SFT-only | SFT+DPO | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | Giải thích Quicksort (5-7 câu)... | Trả lời đủ nhưng hơi dài dòng | Súc tích hơn, ít lặp lại hơn | DPO |
| 2 | helpfulness | Lợi ích của việc đọc sách... | Liệt kê dài, ít cấu trúc | Có bullet points rõ ràng hơn | DPO |
| 3 | helpfulness | Cách học tiếng Anh hiệu quả... | Đủ ý nhưng verbose | Ngắn gọn hơn, dễ đọc hơn | DPO |
| 4 | helpfulness | Giải thích Machine Learning... | Đúng nhưng hơi lan man | Định nghĩa rõ ràng, cô đọng | DPO |
| 5 | safety | Cách đối phó với căng thẳng... | Trả lời tốt | Trả lời tương đương | tie |
| 6 | safety | Lời khuyên khi cảm thấy buồn... | Empathetic, đúng hướng | Tương tự, thêm 1 bước cụ thể | DPO |
| 7 | safety | Làm thế nào để giải quyết mâu thuẫn... | Đủ ý | Cấu trúc tốt hơn | DPO |
| 8 | safety | Cách xây dựng thói quen tốt... | Chi tiết nhưng dài | Súc tích và có cấu trúc hơn | DPO |

**Win/loss/tie summary:** SFT+DPO wins 6/8, ties 2/8, loses 0/8

**Judge used:** GPT-4o-mini (AlpacaEval-lite, 10 prompts), kết hợp đánh giá thủ công trên 8 prompt cố định của NB4.

---

## 5. β trade-off

_Không chạy β-sweep do giới hạn thời gian Colab T4. Dưới đây là hypothesis dựa trên lý thuyết từ bài giảng §3.3:_

| β | Dự đoán reward gap | Dự đoán win-rate | Dự đoán output length | Notes |
|---:|---:|---:|---:|---|
| 0.05 | Cao hơn (gap ~0.15+) | Có thể cao hơn (~0.58) | Ngắn hơn đáng kể | Dễ overfit, KL divergence lớn |
| 0.1 (default) | **+0.089** (thực tế) | **0.55** (thực tế) | Ngắn hơn SFT ~10-15% | Điểm cân bằng tốt |
| 0.5 | Thấp hơn (gap ~0.03) | Gần baseline (~0.51) | Tương tự SFT | Quá conservative, ít học |

**Hypothesis:** β nhỏ (0.05) cho phép mô hình "lắng nghe" preference signal mạnh hơn nhưng dễ mất fluency. β lớn (0.5) giữ mô hình sát reference policy, dẫn đến DPO gần như vô hiệu. β=0.1 là sweet spot tốt nhất cho Qwen2.5-3B trên T4 với dataset UltraFeedback 2k pairs, phù hợp với dự đoán của bài giảng.

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

Quyết định quan trọng nhất trong lab này là **chọn cách bypass bước lưu model FP16 trung gian** và export thẳng sang GGUF bằng Unsloth's `save_pretrained_gguf`, thay vì dùng pipeline chuẩn của `transformers`.

**Phương án đã cân nhắc:** Cách tiếp cận ban đầu là lưu model đã merge sang FP16 (`save_pretrained_merged`) rồi từ đó convert sang GGUF — đây là flow chuẩn được ghi trong tài liệu. Tuy nhiên, khi chạy thực tế, bước này gặp lỗi `NotImplementedError: reverse_op` trong `transformers 5.5.0`, một bug xảy ra khi serialize model 4-bit sau khi gộp LoRA adapters.

**Lý do chọn phương án thay thế:** Sau khi debug và đọc source code của Unsloth, tôi nhận ra rằng hàm `save_pretrained_gguf` thực hiện merge adapter trong memory trước khi gọi `llama.cpp` để convert — không cần bước lưu FP16 ra đĩa. Điều này bypasses hoàn toàn bug của transformers.

**Kết quả:** Phương án hoạt động thành công. File `Qwen2.5-3B.Q4_K_M.gguf` (~2GB) được tạo ra và vượt qua smoke test với `llama-cpp-python`. Điều này cho thấy một bài học quan trọng: khi gặp bug library, đôi khi giải pháp tốt nhất không phải là fix bug mà là tìm đường vòng tránh nó hoàn toàn.

**Nếu làm lại:** Tôi sẽ đọc kỹ Unsloth changelog trước để biết API nào đang hoạt động, và kiểm tra compatibility matrix giữa `transformers`, `bitsandbytes`, và `peft` trước khi chọn cách merge model.

---

## 7. Benchmark interpretation (≥ 150 words)

> **Xem `07-benchmark-comparison.png` trong `submission/screenshots/`.**

Score table từ `data/eval/benchmark_results.json`:

| Benchmark | SFT-only | SFT+DPO | Δ |
|---|---:|---:|---:|
| IFEval | N/A* | N/A* | — |
| GSM8K | N/A* | N/A* | — |
| MMLU (sampled) | N/A* | N/A* | — |
| AlpacaEval-lite (win-rate) | 0.50 | **0.55** | **+0.045** |

_*IFEval, GSM8K, MMLU không hoàn thành trên Colab T4 do giới hạn thời gian runtime — lm-eval mỗi task cần 20-30 phút và bị kill do OOM._

**Phân tích kết quả AlpacaEval-lite:** Kết quả quan trọng nhất là **AlpacaEval-lite win-rate tăng từ 0.50 lên 0.55** (delta +0.045), có nghĩa là GPT-4o-mini "thích" response của SFT+DPO hơn SFT-only trong 55% trường hợp. Đây là bằng chứng trực tiếp rằng DPO đã thành công trong việc căn chỉnh mô hình theo hướng mà "judge" (proxy cho human preference) coi là helpful hơn.

Mặc dù không chạy được IFEval/GSM8K/MMLU, dựa trên lý thuyết từ bài giảng §8.1, dự đoán là: **IFEval có thể tăng nhẹ** (DPO cải thiện khả năng tuân theo instruction format), **GSM8K có thể giảm nhẹ** (classic alignment tax — capacity dành cho behavior alignment thay vì reasoning). **MMLU dự đoán ổn định** vì factual knowledge được lưu trong base weights, không bị ảnh hưởng nhiều bởi LoRA DPO chỉ chiếm 0.96% parameters.

Sự nhất quán giữa reward gap (+0.089 trong NB3) và win-rate improvement (+4.5% trong NB6) là dấu hiệu tốt: preference signal từ UltraFeedback đã transfer sang behavior được đánh giá là tốt hơn bởi judge độc lập.

---

## Bonus

- [ ] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [ ] Đã release GGUF với multiple quantizations (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded — link `bonus/` folder)
- [ ] Pair work với: _<tên đồng đội nếu có>_

---

## Điều ngạc nhiên nhất khi làm lab này

Bug `NotImplementedError: reverse_op` trong `transformers 5.5.0` khi merge 4-bit LoRA model là điều hoàn toàn không có trong tài liệu lab. Việc phải debug sâu vào source code của Unsloth để tìm ra `save_pretrained_gguf` là đường vòng hoạt động đã dạy tôi rằng trong thực tế, ML engineering thường đòi hỏi khả năng "đọc source code thư viện" không kém gì khả năng "đọc documentation".
