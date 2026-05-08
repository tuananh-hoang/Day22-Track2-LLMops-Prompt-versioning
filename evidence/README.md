# 📊 Day 22 Lab Evidence - LangSmith & Prompt Versioning
> 2A202600075 - Hoàng Tuấn Anh <br>
Dưới đây là tổng hợp các minh chứng cho việc triển khai hệ thống Production-grade RAG Pipeline.

## 🔗 LangSmith Public Link
Mọi hoạt động tracing, prompt versioning và evaluation đều có thể xem trực tiếp tại đây:
👉 **(https://smith.langchain.com/public/e7fc52aa-b56e-483b-8d36-f921c5bb0b9d/r)**

---

## 📂 Evidence Checklist
- [x] `01_langsmith_traces.png`: Minh chứng ≥ 50 traces ban đầu.
- [x] `02_prompt_hub.png`: Minh chứng 2 phiên bản prompt trên Hub.
- [x] `02_ab_routing_log.txt`: Nhật ký phân luồng A/B Testing (V1=19, V2=31).
- [x] `03_ragas_scores.png`: Bảng điểm đánh giá từ RAGAS.
- [x] `03_ragas_report.json`: Dữ liệu chi tiết điểm RAGAS.
- [x] `04_pii_demo_log.txt`: Kết quả kiểm duyệt dữ liệu nhạy cảm (PII).
- [x] `04_json_demo_log.txt`: Kết quả sửa lỗi định dạng JSON.

---

## 📈 V1 vs V2 Evaluation Analysis (Bonus)

Dựa trên kết quả từ Step 3 (RAGAS), cả hai phiên bản Prompt đều đạt kết quả rất cao (Faithfulness > 0.9), đạt mục tiêu Bonus của bài Lab.

| Metric | Prompt V1 (Concise) | Prompt V2 (Structured) | Winner |
| :--- | :--- | :--- | :--- |
| **Faithfulness** | 0.9010 | **0.9237** | **V2** |
| **Answer Relevancy** | 0.8369 | **0.8389** | **V2** |
| **Context Recall** | 0.9400 | 0.9400 | Hòa |
| **Context Precision** | 0.6783 | 0.6783 | Hòa |

### Phân tích chi tiết:
1. **Faithfulness:** Phiên bản **V2** đạt điểm cao hơn nhờ vào các chỉ dẫn cụ thể (Structured Instructions). Việc yêu cầu model "trích xuất thông tin kỹ thuật chi tiết từ ngữ cảnh" giúp giảm thiểu tình trạng model tự suy luận (hallucination) so với V1 vốn ưu tiên sự ngắn gọn.
2. **Answer Relevancy:** V2 nhỉnh hơn một chút vì các câu trả lời có cấu trúc giúp bao quát đầy đủ các khía cạnh của câu hỏi hơn là kiểu trả lời trực diện của V1.
3. **Retrieval Metrics:** `Context Recall` và `Context Precision` bằng nhau ở cả 2 phiên bản vì chúng dùng chung một bộ Retriever (FAISS) và cùng kho dữ liệu `knowledge_base.txt`.

## 🛡️ Guardrails AI Results
- **PII Detector:** Đã nhận diện chính xác và đánh dấu (Redact) các thành phần nhạy cảm như Email, Phone, SSN và Credit Card.
- **JSON Validator:** Đã sửa lỗi thành công cho các trường hợp Markdown fences và trailing commas, đảm bảo đầu ra luôn sẵn sàng cho ứng dụng phía sau.
