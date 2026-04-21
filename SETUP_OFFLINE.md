# 🔄 TF-IDF Offline Setup Guide

## ✅ Current Configuration: TF-IDF-Only Mode (No HuggingFace Required)

Your project is now configured to use **TF-IDF-based retrieval**, which works entirely offline without requiring access to HuggingFace.

### What Changed:
- `USE_TFIDF_ONLY=true` is set in `.env` (default)
- The Streamlit app will use TF-IDF retrieval by default
- No network access to HuggingFace is required
- You can still upload PDFs and get search results

---

## 🚀 Quick Start

### Run the Streamlit App
```bash
streamlit run app/streamlit_app.py
```

The app will:
1. Show **TF-IDF-only mode** as the default in Configuration
2. Use keyword-based TF-IDF retrieval for all document searches
3. Work completely offline without HuggingFace

### Index Documents with TF-IDF
```bash
python scripts/index_documents.py --data_dir ./data --index_dir ./index
```

The indexing script will:
1. Process all PDFs in `data/` directory
2. Build a TF-IDF index (no ColPali embeddings)
3. Save indexes and page images to `index/` directory

---

## 📊 TF-IDF vs ColPali Comparison

| Aspect | TF-IDF | ColPali |
|--------|--------|---------|
| **Internet Required** | ❌ No | ✅ Yes (HuggingFace) |
| **Model Size** | Small | ~2GB |
| **Speed** | ⚡ Fast | Slower |
| **Accuracy** | Good for keywords | Excellent (visual understanding) |
| **Best For** | Offline, keyword queries | Complex semantic searches |

---

## 🔧 Configuration Options

### Option 1: Use TF-IDF Only (Current - Default)
```
# In .env file:
USE_TFIDF_ONLY=true
```
✅ **This is your current setup** - No HuggingFace needed

### Option 2: Switch Back to ColPali (When HuggingFace is Available)
```
# In .env file:
USE_TFIDF_ONLY=false
```
Then run:
```bash
streamlit run app/streamlit_app.py
```
The app will download the ColPali model from HuggingFace on first run.

### Option 3: Use Local ColPali Model
If you have a ColPali checkpoint locally:
```
# In .env file:
USE_TFIDF_ONLY=false
COLPALI_MODEL=/path/to/local/model
```

### Option 4: Force ColPali via UI (Auto Mode)
1. Open the Streamlit app
2. In Configuration → Retrieval mode, select **"Auto (ColPali if available)"**
3. The app will try to load ColPali, fallback to TF-IDF if HuggingFace is unreachable

---

## 📁 File Structure

```
.env                          ← Configuration (USE_TFIDF_ONLY=true)
app/
  streamlit_app.py           ← Main app (now uses TF-IDF by default)
src/
  ingestion/
    embedder.py              ← Now detects TF-IDF mode
    indexer.py               ← TextPageIndex for TF-IDF
  retrieval/
    text_retriever.py        ← TF-IDF retriever
    retriever.py             ← ColPali retriever (optional)
scripts/
  index_documents.py         ← Indexing script (TF-IDF by default)
data/                        ← Your PDF documents
index/                       ← Saved TF-IDF indexes
```

---

## ⚠️ Important Notes

### TF-IDF Limitations:
- Uses **keyword matching** only (no semantic understanding)
- May miss relevant pages if they use different terminology
- Works best with clear, searchable text

### ColPali Advantages:
- Understands **visual content** (charts, diagrams, layout)
- Better for complex semantic queries
- More accurate but requires HuggingFace access

---

## 🔄 Switching Modes Later

If you later get access to HuggingFace:

1. **Edit `.env`:**
   ```
   USE_TFIDF_ONLY=false
   ```

2. **Run the app:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

3. **First run will download ColPali** (~2GB)

4. **Reindex documents** with ColPali (optional):
   ```bash
   python scripts/index_documents.py --data_dir ./data --index_dir ./index
   ```

---

## ✅ Verification Checklist

- [ ] `.env` has `USE_TFIDF_ONLY=true`
- [ ] Streamlit app starts without errors
- [ ] Can upload PDFs in the app
- [ ] Can retrieve search results
- [ ] No HuggingFace download attempts
- [ ] Indexing script works with TF-IDF mode

---

## 📞 Troubleshooting

### Error: "Failed to load ColPali model"
→ This is expected! You're using TF-IDF mode. Check that `USE_TFIDF_ONLY=true` in `.env`

### App shows "Auto (ColPali if available)"
→ The UI is set to Auto mode. Change to "Force TF-IDF only" in the Configuration sidebar

### TF-IDF index not found
→ Run: `python scripts/index_documents.py --data_dir ./data --index_dir ./index`

### Slow retrieval
→ TF-IDF can be slow on large documents. Consider:
   - Reducing DPI when uploading PDFs (use 72-100 instead of 150)
   - Limiting number of pages per PDF
   - Using smaller documents

---

## 📚 What is TF-IDF?

**TF-IDF** (Term Frequency-Inverse Document Frequency) is a classic information retrieval technique that:
- Counts word occurrences in documents (Term Frequency)
- Weights rare/important words higher (Inverse Document Frequency)
- Scores documents by word overlap with your query
- No neural networks, no internet required

Perfect for **offline, keyword-based document retrieval**!

---

## 🎯 Next Steps

1. ✅ **Verify setup:** Run the Streamlit app
2. 📄 **Upload PDFs:** Test with your documents
3. 🔍 **Try searches:** Search for keywords in your documents
4. 📊 **Index your data:** Run `index_documents.py` with your PDFs

Enjoy your offline document intelligence system! 🚀
