"""

Tra cứu thông tin kanji từ DB nội bộ, Jisho API và Gemini AI.

Trả về: âm on/kun, nghĩa tiếng Việt, mẹo nhớ, từ vựng ví dụ.

"""

import json

import os

import sys

import threading



GEMINI_MODEL = "gemini-2.0-flash"  # free tier trên Google AI Studio

OPENROUTER_MODEL        = "nvidia/nemotron-3-super-120b-a12b:free"  # free, không tốn credit
OPENROUTER_ANALYZE_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"   # nhỏ hơn, nhanh hơn cho phân tích

import unicodedata

import requests





def _get_app_dir() -> str:

    """Trả về thư mục chứa EXE (khi frozen) hoặc thư mục source (khi dev)."""

    if getattr(sys, "frozen", False):

        return os.path.dirname(sys.executable)

    return os.path.dirname(os.path.abspath(__file__))





# ─── Lookup cache (persistent) ──────────────────────────────────────────────────

_CACHE_PATH = os.path.join(_get_app_dir(), "lookup_cache.json")

_cache_lock = threading.Lock()

_mem_cache: dict = {}   # in-memory cache: kanji → info dict



def _load_cache():

    global _mem_cache

    try:

        with open(_CACHE_PATH, encoding="utf-8") as f:

            _mem_cache = json.load(f)

    except Exception:

        _mem_cache = {}



def _save_cache():

    with _cache_lock:

        try:

            with open(_CACHE_PATH, "w", encoding="utf-8") as f:

                json.dump(_mem_cache, f, ensure_ascii=False, indent=2)

        except Exception:

            pass



_load_cache()



# ─── Config ───────────────────────────────────────────────────────────────────

_CONFIG_PATH = os.path.join(_get_app_dir(), "config.json")


def _init_config_from_bundle():
    """Khi chạy dưới dạng EXE: nếu chưa có config.json cạnh exe,
    copy từ _internal/ (bundled defaults có API key) ra thư mục exe."""
    if not getattr(sys, "frozen", False):
        return
    if os.path.exists(_CONFIG_PATH):
        return
    bundled = os.path.join(sys._MEIPASS, "config.json")
    if os.path.exists(bundled):
        import shutil
        shutil.copy2(bundled, _CONFIG_PATH)


_init_config_from_bundle()


def load_config() -> dict:

    try:

        with open(_CONFIG_PATH, encoding="utf-8") as f:

            return json.load(f)

    except Exception:

        return {}


def save_config(data: dict):

    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:

        json.dump(data, f, ensure_ascii=False, indent=4)




def get_gemini_key() -> str:

    return load_config().get("gemini_api_key", "").strip()





def set_gemini_key(key: str):

    cfg = load_config()

    cfg["gemini_api_key"] = key.strip()

    save_config(cfg)





def get_openrouter_key() -> str:

    return load_config().get("openrouter_api_key", "").strip()



def set_openrouter_key(key: str):

    cfg = load_config()

    cfg["openrouter_api_key"] = key.strip()

    save_config(cfg)



def get_ai_provider() -> str:

    """Trả về 'gemini' hoặc 'openrouter'."""

    return load_config().get("ai_provider", "gemini")



def set_ai_provider(provider: str):

    cfg = load_config()

    cfg["ai_provider"] = provider

    save_config(cfg)



class AIQuotaError(Exception):

    """Hết quota AI."""

    pass



class GeminiQuotaError(AIQuotaError):

    pass



JISHO_API = "https://jisho.org/api/v1/search/words?keyword={}"



_GEMINI_PROMPT = """

Bạn là chuyên gia dạy tiếng Nhật cho người Việt Nam.

Hãy tra cứu kanji "{kanji}" và trả về JSON với format CHÍNH XÁC sau (không giải thích thêm):

{{

  "viet": "ÂM HÁN VIỆT (chữ HOA)",

  "reading": "cách đọc hiragana phổ biến nhất",

  "meaning_vi": "nghĩa tiếng Việt ngắn gọn",

  "meo": "mẹo nhớ vui bằng tiếng Việt liên quan hình dạng chữ (để trống nếu không có)",

  "vocab": [

    ["từ_kanji_1", "hiragana_1", "nghĩa_việt_1"],

    ["từ_kanji_2", "hiragana_2", "nghĩa_việt_2"]

  ]

}}

"""



_ANALYZE_PROMPT = """

Bạn là chuyên gia ngôn ngữ học và giảng viên tiếng Nhật cao cấp.

Hãy phân tích CHI TIẾT chữ Hán "{kanji}" cho người học tiếng Việt.

Yêu cầu nội dung phản hồi bằng tiếng Việt, trình bày đẹp, sử dụng các emoji để phân biệt các mục.



Cấu trúc nội dung:

📖 Chữ: {kanji} (âm đọc chính)

🔠 Hán Việt: [Âm Hán Việt]

📝 Nghĩa: [Giải thích nghĩa chi tiết]



💡 Cách dùng và Ví dụ:

- Liệt kê 3 ví dụ thực tế nhất.

- Mỗi ví dụ gồm: Câu tiếng Nhật (có kanji) -> Phiên âm Hiragana -> Dịch nghĩa tiếng Việt.



⚠️ Lưu ý quan trọng:

- Giải thích các cách đọc biến âm, âm On/Kun cần chú ý của chữ này trong các ngữ cảnh khác nhau.

- Các mẹo nhỏ để phân biệt với các chữ gần giống (nếu có).



Hãy trả về văn bản thuần túy (không cần định dạng code block), trình bày thoáng đãng.

"""





def lookup_kanji_gemini(kanji: str) -> dict:

    """Dùng Gemini AI để tra kanji, trả kết quả tiếng Việt đầy đủ."""

    api_key = get_gemini_key()

    if not api_key:

        return {}

    try:

        from google import genai

        client = genai.Client(api_key=api_key)

        prompt = _GEMINI_PROMPT.format(kanji=kanji)

        response = client.models.generate_content(

            model=GEMINI_MODEL,

            contents=prompt,

        )

        raw = response.text.strip()

        # Bỏ markdown code block nếu có

        if raw.startswith("```"):

            raw = raw.split("\n", 1)[-1]

            raw = raw.rsplit("```", 1)[0]

        data = json.loads(raw)

        vocab_raw = data.get("vocab", [])

        vocab = [tuple(v) for v in vocab_raw if len(v) >= 3]

        return {

            "viet":       data.get("viet", ""),

            "reading":    data.get("reading", ""),

            "meaning_vi": data.get("meaning_vi", ""),

            "meo":        data.get("meo", ""),

            "vocab":      vocab[:2],

        }

    except Exception as e:

        msg = str(e)

        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:

            raise GeminiQuotaError("Hết quota Gemini hôm nay, vui lòng thử lại vào ngày mai.")

        print(f"[gemini] Lỗi khi tra {kanji}: {e}")

        return {}





def lookup_kanji_openrouter(kanji: str) -> dict:

    """Dùng OpenRouter API để tra kanji."""

    api_key = get_openrouter_key()

    if not api_key:

        return {}

    

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {

        "Authorization": f"Bearer {api_key}",

        "Content-Type": "application/json",

        "HTTP-Referer": "https://github.com/antigravity", # Thay bằng URL app của bạn nếu có

        "X-Title": "Kanji Hub"

    }

    

    prompt = _GEMINI_PROMPT.format(kanji=kanji)

    payload = {

        "model": OPENROUTER_MODEL,

        "messages": [{"role": "user", "content": prompt}]

    }

    

    try:

        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.encoding = 'utf-8'
        if resp.status_code in (401, 403):

            raise AIQuotaError("Lỗi xác thực OpenRouter (API Key sai hoặc hết hạn).")

        if resp.status_code == 429:

            raise AIQuotaError("Hết quota OpenRouter hoặc bị limit. Thử lại sau.")

        

        resp_data = resp.json()

        if "choices" not in resp_data:

            print(f"[openrouter] Lỗi: {resp_data}")

            return {}

            

        raw = resp_data["choices"][0]["message"]["content"].strip()

        

        # Bóc tách JSON an toàn hơn

        if "{" in raw:

            raw = raw[raw.find("{"):raw.rfind("}")+1]

            

        data = json.loads(raw)

        vocab_raw = data.get("vocab", [])

        vocab = [tuple(v) for v in vocab_raw if len(v) >= 3]

        return {

            "viet":       data.get("viet", ""),

            "reading":    data.get("reading", ""),

            "meaning_vi": data.get("meaning_vi", ""),

            "meo":        data.get("meo", ""),

            "vocab":      vocab[:2],

        }

    except AIQuotaError:

        raise

    except Exception as e:

        print(f"[openrouter] Lỗi khi tra {kanji}: {e}")

        return {}





def lookup_kanji_mazii(kanji: str) -> dict:

    """Dùng Mazii API để tra kanji (Nhật-Việt)."""

    try:

        url = "https://mazii.net/api/search"

        payload = {"dict": "javi", "type": "kanji", "query": kanji}

        resp = requests.post(url, json=payload, timeout=10)

        if resp.status_code != 200:

            return {}

        

        data = resp.json()

        results = data.get("results", [])

        if not results:

            return {}

            

        item = results[0]

        # Xử lý vocab từ examples của Mazii

        vocab = []

        for ex in item.get("examples", [])[:2]:

            vocab.append((ex.get("w", ""), ex.get("p", "").strip(), ex.get("m", "")))

            

        return {

            "viet":       item.get("mean", ""),

            "reading":    item.get("kun", "") or item.get("on", ""),

            "meaning_vi": item.get("detail", "").replace("##", "\n").split("\n")[0].strip(), # Lấy nghĩa đầu tiên cho gọn

            "meo":        "", 

            "vocab":      vocab,

            "source":     "mazii"

        }

    except Exception as e:

        print(f"[mazii] Lỗi khi tra {kanji}: {e}")

        return {}





def analyze_kanji_ai(kanji: str) -> str:

    """Dùng AI (Gemini hoặc OpenRouter) để phân tích chuyên sâu."""

    provider = get_ai_provider()

    api_key = get_gemini_key() if provider == "gemini" else get_openrouter_key()

    if not api_key:

        return "⚠ Chưa cài đặt API Key cho nhà cung cấp AI hiện tại."

    

    prompt = _ANALYZE_PROMPT.format(kanji=kanji)

    

    if provider == "gemini":

        try:

            from google import genai

            client = genai.Client(api_key=api_key)

            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)

            return response.text.strip()

        except Exception as e:

            return f"✗ Lỗi Gemini: {e}"

    else:

        try:

            url = "https://openrouter.ai/api/v1/chat/completions"

            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

            payload = {

                "model": OPENROUTER_ANALYZE_MODEL,

                "messages": [{"role": "user", "content": prompt}]

            }

            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.encoding = 'utf-8'
            data = resp.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"].strip()
            err = data.get("error", {})
            err_msg = err.get("message", "") if isinstance(err, dict) else str(err)
            return f"✗ OpenRouter: {err_msg or data}"

        except Exception as e:

            return f"✗ Lỗi OpenRouter: {e}"





import urllib.parse



def translate_en_to_vi(text: str) -> str:

    if not text:

        return ""

    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=vi&dt=t&q={urllib.parse.quote(text)}"

    try:

        response = requests.get(url, timeout=3)

        if response.status_code == 200:

            data = response.json()

            return "".join([sentence[0] for sentence in data[0]])

    except Exception:

        pass

    return text



def lookup_kanji_jisho(kanji: str) -> dict:

    """Dùng Jisho API (miễn phí, không cần key)."""

    try:

        url = JISHO_API.format(kanji)

        resp = requests.get(url, timeout=5)

        data = resp.json()

        results = data.get("data", [])

        if not results:

            return {}



        # Lấy entry đầu tiên có kanji khớp chính xác

        for entry in results:

            japanese = entry.get("japanese", [])

            for j in japanese:

                if j.get("word") == kanji or j.get("reading"):

                    senses = entry.get("senses", [{}])

                    meanings = senses[0].get("english_definitions", [])

                    reading = j.get("reading", "")

                    word = j.get("word", kanji)

                    

                    meanings_vi = translate_en_to_vi(", ".join(meanings[:3])) if meanings else ""

                    

                    return {

                        "kanji": kanji,

                        "reading": reading,

                        "meanings_en": meanings[:3],

                        "meaning_vi": meanings_vi,

                    }

        return {}

    except Exception as e:

        print(f"[lookup] Lỗi khi tra {kanji}: {e}")

        return {}





# Bảng nghĩa tiếng Việt thủ công cho các kanji phổ biến (fallback / override)

# vocab: list of (chữ, hiragana, nghĩa_tiếng_việt)

MANUAL_VI: dict[str, dict] = {

    "毛": {"viet": "MAO",    "meaning_vi": "Lông",               "meo": "Lông (毛) ngược với tay (手)",                                    "reading": "け",       "vocab": [("毛糸", "けいと", "sợi len"), ("毛布", "もうふ", "chăn len")]},

    "刀": {"viet": "ĐAO",    "meaning_vi": "Kiếm / Dao",         "meo": "Lực (力) thừa, đao (刀) thụt",                                    "reading": "かたな",   "vocab": [("刀剣", "とうけん", "đao kiếm"), ("小刀", "こがたな", "dao nhỏ")]},

    "力": {"viet": "LỰC",    "meaning_vi": "Sức lực",            "meo": "Đại Ka (カ) phải có Lực (力)",                                    "reading": "ちから",   "vocab": [("力士", "りきし", "võ sĩ sumo"), ("努力", "どりょく", "nỗ lực")]},

    "丸": {"viet": "HOÀN",   "meaning_vi": "Tròn",               "meo": "Số 9 (九) cắt (、) đuôi thì hoàn toàn là hình tròn (丸)",        "reading": "まる",     "vocab": [("丸い", "まるい", "hình tròn"), ("丸太", "まるた", "khúc gỗ")]},

    "究": {"viet": "CỨU",    "meaning_vi": "Nghiên cứu",         "meo": "Đào lỗ (穴) xuống cửu (九) tuyền để nghiên cứu (究)",            "reading": "きわめる", "vocab": [("研究", "けんきゅう", "nghiên cứu"), ("究明", "きゅうめい", "làm rõ")]},

    "酒": {"viet": "TỬU",    "meaning_vi": "Rượu",               "meo": "Ủ men với nước (氵) tới giờ Dậu (酉) sẽ thành rượu (酒)",        "reading": "さけ",     "vocab": [("日本酒", "にほんしゅ", "rượu sake"), ("お酒", "おさけ", "rượu")]},

    "光": {"viet": "QUANG",  "meaning_vi": "Ánh sáng",           "meo": "Tiểu (小) cô nương có 1 (一) cặp chân (儿) trắng sáng (光)",     "reading": "ひかり",   "vocab": [("光る", "ひかる", "sáng lấp lánh"), ("日光", "にっこう", "ánh nắng")]},

    "当": {"viet": "ĐƯƠNG",  "meaning_vi": "Đương thời, hiện tại","meo": "Tiểu (小) cô nương! Em (ヨ) đương nhiên (当) rất xinh đẹp!",   "reading": "あたる",   "vocab": [("当然", "とうぜん", "đương nhiên"), ("担当", "たんとう", "phụ trách")]},

    "社": {"viet": "XÃ",     "meaning_vi": "Xã hội, công ty",    "meo": "Chỉ thị (ネ) cúng thổ (土) địa cho xã hội (社) phát triển",     "reading": "しゃ",     "vocab": [("会社", "かいしゃ", "công ty"), ("社会", "しゃかい", "xã hội")]},

    "降": {"viet": "GIÁNG",  "meaning_vi": "Xuống, rơi",         "meo": "Bồ (阝) sau (夂) 14 (牛) năm bị giáng (降) chức",               "reading": "おりる",   "vocab": [("降る", "ふる", "mưa/tuyết rơi"), ("下降", "かこう", "đi xuống")]},

}



# ── N4 Kanji (167 chữ) ───────────────────────────────────────────────────────

N4_VI: dict[str, dict] = {

    "会": {"viet": "HỘI",    "meaning_vi": "Gặp gỡ, hội họp",        "meo": "", "reading": "あう",     "vocab": [("会社", "かいしゃ", "công ty"),          ("会議", "かいぎ", "cuộc họp")]},

    "同": {"viet": "ĐỒNG",   "meaning_vi": "Giống nhau, cùng",        "meo": "", "reading": "おなじ",   "vocab": [("同じ", "おなじ", "giống nhau"),         ("同時", "どうじ", "đồng thời")]},

    "事": {"viet": "SỰ",     "meaning_vi": "Việc, sự việc",           "meo": "", "reading": "こと",     "vocab": [("仕事", "しごと", "công việc"),          ("事故", "じこ", "tai nạn")]},

    "自": {"viet": "TỰ",     "meaning_vi": "Tự mình, bản thân",       "meo": "", "reading": "じぶん",   "vocab": [("自分", "じぶん", "bản thân"),           ("自転車", "じてんしゃ", "xe đạp")]},

    "発": {"viet": "PHÁT",   "meaning_vi": "Xuất phát, phát ra",      "meo": "", "reading": "はつ",     "vocab": [("出発", "しゅっぱつ", "khởi hành"),      ("発表", "はっぴょう", "công bố")]},

    "者": {"viet": "GIẢ",    "meaning_vi": "Người, kẻ",               "meo": "", "reading": "もの",     "vocab": [("医者", "いしゃ", "bác sĩ"),             ("学者", "がくしゃ", "học giả")]},

    "地": {"viet": "ĐỊA",    "meaning_vi": "Đất, mặt đất",            "meo": "", "reading": "ち",       "vocab": [("地図", "ちず", "bản đồ"),               ("地下鉄", "ちかてつ", "tàu điện ngầm")]},

    "業": {"viet": "NGHIỆP", "meaning_vi": "Nghề nghiệp, công việc",  "meo": "", "reading": "ぎょう",   "vocab": [("授業", "じゅぎょう", "buổi học"),       ("工業", "こうぎょう", "công nghiệp")]},

    "方": {"viet": "PHƯƠNG", "meaning_vi": "Hướng, phía, cách",       "meo": "", "reading": "かた",     "vocab": [("方法", "ほうほう", "phương pháp"),      ("地方", "ちほう", "địa phương")]},

    "新": {"viet": "TÂN",    "meaning_vi": "Mới",                     "meo": "", "reading": "あたらしい","vocab": [("新聞", "しんぶん", "báo"),               ("新幹線", "しんかんせん", "tàu cao tốc")]},

    "場": {"viet": "TRƯỜNG", "meaning_vi": "Địa điểm, nơi chốn",      "meo": "", "reading": "ば",       "vocab": [("場所", "ばしょ", "địa điểm"),           ("工場", "こうじょう", "nhà máy")]},

    "員": {"viet": "VIÊN",   "meaning_vi": "Nhân viên, thành viên",   "meo": "", "reading": "いん",     "vocab": [("店員", "てんいん", "nhân viên cửa hàng"),("会員", "かいいん", "hội viên")]},

    "立": {"viet": "LẬP",    "meaning_vi": "Đứng, thành lập",         "meo": "", "reading": "たつ",     "vocab": [("立つ", "たつ", "đứng dậy"),             ("役立つ", "やくだつ", "có ích")]},

    "開": {"viet": "KHAI",   "meaning_vi": "Mở ra, khai mạc",         "meo": "", "reading": "あける",   "vocab": [("開く", "ひらく", "mở ra"),              ("開店", "かいてん", "khai trương")]},

    "手": {"viet": "THỦ",    "meaning_vi": "Tay",                     "meo": "", "reading": "て",       "vocab": [("手紙", "てがみ", "thư"),                ("上手", "じょうず", "giỏi")]},

    "問": {"viet": "VẤN",    "meaning_vi": "Câu hỏi, vấn đề",         "meo": "", "reading": "とう",     "vocab": [("問題", "もんだい", "vấn đề"),           ("質問", "しつもん", "câu hỏi")]},

    "代": {"viet": "ĐẠI",    "meaning_vi": "Thay thế, thời đại",      "meo": "", "reading": "かわり",   "vocab": [("時代", "じだい", "thời đại"),           ("代わり", "かわり", "thay thế")]},

    "明": {"viet": "MINH",   "meaning_vi": "Sáng, rõ ràng",           "meo": "", "reading": "あかるい", "vocab": [("説明", "せつめい", "giải thích"),       ("明日", "あした", "ngày mai")]},

    "動": {"viet": "ĐỘNG",   "meaning_vi": "Di chuyển, động",         "meo": "", "reading": "うごく",   "vocab": [("運動", "うんどう", "vận động"),         ("動物", "どうぶつ", "động vật")]},

    "京": {"viet": "KINH",   "meaning_vi": "Kinh đô, thủ đô",         "meo": "", "reading": "きょう",   "vocab": [("東京", "とうきょう", "Tokyo"),          ("京都", "きょうと", "Kyoto")]},

    "目": {"viet": "MỤC",    "meaning_vi": "Mắt",                     "meo": "", "reading": "め",       "vocab": [("目的", "もくてき", "mục đích"),         ("注目", "ちゅうもく", "chú ý")]},

    "通": {"viet": "THÔNG",  "meaning_vi": "Thông qua, đi lại",       "meo": "", "reading": "とおる",   "vocab": [("交通", "こうつう", "giao thông"),       ("通う", "かよう", "đi lại")]},

    "言": {"viet": "NGÔN",   "meaning_vi": "Nói, lời nói",            "meo": "", "reading": "いう",     "vocab": [("言葉", "ことば", "từ ngữ"),             ("言語", "げんご", "ngôn ngữ")]},

    "理": {"viet": "LÝ",     "meaning_vi": "Lý lẽ, lý do",            "meo": "", "reading": "り",       "vocab": [("料理", "りょうり", "nấu ăn"),           ("理由", "りゆう", "lý do")]},

    "体": {"viet": "THỂ",    "meaning_vi": "Cơ thể, thân thể",        "meo": "", "reading": "からだ",   "vocab": [("体育", "たいいく", "thể dục"),          ("体験", "たいけん", "trải nghiệm")]},

    "田": {"viet": "ĐIỀN",   "meaning_vi": "Ruộng lúa",               "meo": "", "reading": "た",       "vocab": [("田んぼ", "たんぼ", "ruộng lúa"),        ("田舎", "いなか", "nông thôn")]},

    "主": {"viet": "CHỦ",    "meaning_vi": "Chủ, người chủ",          "meo": "", "reading": "ぬし",     "vocab": [("主人", "しゅじん", "chủ nhà/chồng"),    ("主語", "しゅご", "chủ ngữ")]},

    "題": {"viet": "ĐỀ",     "meaning_vi": "Đề tài, chủ đề",          "meo": "", "reading": "だい",     "vocab": [("問題", "もんだい", "vấn đề"),           ("宿題", "しゅくだい", "bài tập về nhà")]},

    "意": {"viet": "Ý",      "meaning_vi": "Ý nghĩa, tâm ý",          "meo": "", "reading": "い",       "vocab": [("意味", "いみ", "ý nghĩa"),              ("注意", "ちゅうい", "chú ý")]},

    "不": {"viet": "BẤT",    "meaning_vi": "Không, bất (phủ định)",   "meo": "", "reading": "ふ",       "vocab": [("不便", "ふべん", "bất tiện"),           ("不安", "ふあん", "lo lắng")]},

    "作": {"viet": "TÁC",    "meaning_vi": "Làm, sản xuất",           "meo": "", "reading": "つくる",   "vocab": [("作文", "さくぶん", "bài luận"),         ("作品", "さくひん", "tác phẩm")]},

    "用": {"viet": "DỤNG",   "meaning_vi": "Sử dụng, dùng",           "meo": "", "reading": "よう",     "vocab": [("用意", "ようい", "chuẩn bị"),           ("使用", "しよう", "sử dụng")]},

    "度": {"viet": "ĐỘ",     "meaning_vi": "Độ, lần, mức độ",         "meo": "", "reading": "ど",       "vocab": [("今度", "こんど", "lần này/tới"),        ("温度", "おんど", "nhiệt độ")]},

    "強": {"viet": "CƯỜNG",  "meaning_vi": "Mạnh, cứng rắn",          "meo": "", "reading": "つよい",   "vocab": [("勉強", "べんきょう", "học tập"),        ("強い", "つよい", "mạnh")]},

    "公": {"viet": "CÔNG",   "meaning_vi": "Công cộng, công khai",    "meo": "", "reading": "こう",     "vocab": [("公園", "こうえん", "công viên"),        ("公共", "こうきょう", "công cộng")]},

    "持": {"viet": "TRÌ",    "meaning_vi": "Cầm, giữ, có",            "meo": "", "reading": "もつ",     "vocab": [("持つ", "もつ", "cầm, giữ"),             ("気持ち", "きもち", "cảm giác")]},

    "野": {"viet": "DÃ",     "meaning_vi": "Đồng ruộng, ngoài trời", "meo": "", "reading": "の",       "vocab": [("野菜", "やさい", "rau củ"),             ("野球", "やきゅう", "bóng chày")]},

    "以": {"viet": "DĨ",     "meaning_vi": "Bằng cách, với",          "meo": "", "reading": "い",       "vocab": [("以上", "いじょう", "trên, hơn"),        ("以下", "いか", "dưới, ít hơn")]},

    "思": {"viet": "TƯ",     "meaning_vi": "Nghĩ, suy nghĩ",          "meo": "", "reading": "おもう",   "vocab": [("思う", "おもう", "nghĩ"),               ("思い出", "おもいで", "kỷ niệm")]},

    "家": {"viet": "GIA",    "meaning_vi": "Nhà, gia đình",            "meo": "", "reading": "いえ",     "vocab": [("家族", "かぞく", "gia đình"),           ("家具", "かぐ", "đồ nội thất")]},

    "世": {"viet": "THẾ",    "meaning_vi": "Thế giới, thế hệ",        "meo": "", "reading": "よ",       "vocab": [("世界", "せかい", "thế giới"),           ("世話", "せわ", "chăm sóc")]},

    "多": {"viet": "ĐA",     "meaning_vi": "Nhiều",                   "meo": "", "reading": "おおい",   "vocab": [("多い", "おおい", "nhiều"),              ("多分", "たぶん", "có lẽ")]},

    "正": {"viet": "CHÍNH",  "meaning_vi": "Đúng, chính xác",         "meo": "", "reading": "ただしい", "vocab": [("正しい", "ただしい", "đúng đắn"),       ("正直", "しょうじき", "thật thà")]},

    "安": {"viet": "AN",     "meaning_vi": "An toàn, rẻ, bình yên",   "meo": "", "reading": "やすい",   "vocab": [("安い", "やすい", "rẻ"),                 ("安心", "あんしん", "yên tâm")]},

    "院": {"viet": "VIỆN",   "meaning_vi": "Viện, cơ sở, bệnh viện", "meo": "", "reading": "いん",     "vocab": [("病院", "びょういん", "bệnh viện"),      ("大学院", "だいがくいん", "cao học")]},

    "心": {"viet": "TÂM",    "meaning_vi": "Trái tim, tâm trí",       "meo": "", "reading": "こころ",   "vocab": [("心配", "しんぱい", "lo lắng"),          ("安心", "あんしん", "yên tâm")]},

    "界": {"viet": "GIỚI",   "meaning_vi": "Thế giới, giới hạn",      "meo": "", "reading": "かい",     "vocab": [("世界", "せかい", "thế giới"),           ("限界", "げんかい", "giới hạn")]},

    "教": {"viet": "GIÁO",   "meaning_vi": "Dạy học, giáo dục",       "meo": "", "reading": "おしえる", "vocab": [("教室", "きょうしつ", "phòng học"),      ("教科書", "きょうかしょ", "sách giáo khoa")]},

    "文": {"viet": "VĂN",    "meaning_vi": "Văn chương, câu văn",     "meo": "", "reading": "ぶん",     "vocab": [("文章", "ぶんしょう", "đoạn văn"),       ("文化", "ぶんか", "văn hóa")]},

    "元": {"viet": "NGUYÊN", "meaning_vi": "Gốc, nguồn gốc",          "meo": "", "reading": "もと",     "vocab": [("元気", "げんき", "khỏe mạnh"),          ("元々", "もともと", "vốn dĩ")]},

    "重": {"viet": "TRỌNG",  "meaning_vi": "Nặng, quan trọng",        "meo": "", "reading": "おもい",   "vocab": [("重要", "じゅうよう", "quan trọng"),     ("体重", "たいじゅう", "cân nặng")]},

    "近": {"viet": "CẬN",    "meaning_vi": "Gần",                     "meo": "", "reading": "ちかい",   "vocab": [("近く", "ちかく", "gần đây"),            ("最近", "さいきん", "gần đây/dạo này")]},

    "考": {"viet": "KHẢO",   "meaning_vi": "Suy nghĩ, cân nhắc",      "meo": "", "reading": "かんがえる","vocab": [("考える", "かんがえる", "suy nghĩ"),    ("参考", "さんこう", "tham khảo")]},

    "画": {"viet": "HỌA",    "meaning_vi": "Tranh, hình vẽ",          "meo": "", "reading": "え",       "vocab": [("映画", "えいが", "phim"),               ("計画", "けいかく", "kế hoạch")]},

    "海": {"viet": "HẢI",    "meaning_vi": "Biển, đại dương",         "meo": "", "reading": "うみ",     "vocab": [("海外", "かいがい", "nước ngoài"),       ("海水", "かいすい", "nước biển")]},

    "売": {"viet": "MẠI",    "meaning_vi": "Bán",                     "meo": "", "reading": "うる",     "vocab": [("売る", "うる", "bán"),                  ("売り場", "うりば", "quầy bán")]},

    "知": {"viet": "TRI",    "meaning_vi": "Biết, hiểu",              "meo": "", "reading": "しる",     "vocab": [("知る", "しる", "biết"),                 ("知識", "ちしき", "kiến thức")]},

    "道": {"viet": "ĐẠO",    "meaning_vi": "Con đường, đường sá",     "meo": "", "reading": "みち",     "vocab": [("道路", "どうろ", "đường bộ"),           ("歩道", "ほどう", "vỉa hè")]},

    "集": {"viet": "TẬP",    "meaning_vi": "Tập hợp, thu thập",       "meo": "", "reading": "あつめる", "vocab": [("集める", "あつめる", "thu thập"),       ("集合", "しゅうごう", "tập hợp")]},

    "別": {"viet": "BIỆT",   "meaning_vi": "Tách biệt, chia tay",     "meo": "", "reading": "べつ",     "vocab": [("特別", "とくべつ", "đặc biệt"),         ("別に", "べつに", "không hẳn")]},

    "物": {"viet": "VẬT",    "meaning_vi": "Đồ vật, thứ",             "meo": "", "reading": "もの",     "vocab": [("食べ物", "たべもの", "đồ ăn"),          ("動物", "どうぶつ", "động vật")]},

    "使": {"viet": "SỬ",     "meaning_vi": "Sử dụng, dùng",           "meo": "", "reading": "つかう",   "vocab": [("使う", "つかう", "dùng"),               ("大使館", "たいしかん", "đại sứ quán")]},

    "品": {"viet": "PHẨM",   "meaning_vi": "Hàng hóa, phẩm chất",    "meo": "", "reading": "しな",     "vocab": [("品物", "しなもの", "hàng hóa"),         ("食品", "しょくひん", "thực phẩm")]},

    "計": {"viet": "KẾ",     "meaning_vi": "Kế hoạch, đo lường",      "meo": "", "reading": "けい",     "vocab": [("時計", "とけい", "đồng hồ"),            ("計画", "けいかく", "kế hoạch")]},

    "死": {"viet": "TỬ",     "meaning_vi": "Chết",                    "meo": "", "reading": "しぬ",     "vocab": [("死ぬ", "しぬ", "chết"),                 ("死亡", "しぼう", "tử vong")]},

    "特": {"viet": "ĐẶC",    "meaning_vi": "Đặc biệt",                "meo": "", "reading": "とく",     "vocab": [("特別", "とくべつ", "đặc biệt"),         ("特急", "とっきゅう", "tàu đặc tốc")]},

    "私": {"viet": "TƯ",     "meaning_vi": "Tôi, riêng tư",           "meo": "", "reading": "わたし",   "vocab": [("私立", "しりつ", "tư lập"),             ("私鉄", "してつ", "tàu tư nhân")]},

    "始": {"viet": "THỈ",    "meaning_vi": "Bắt đầu",                 "meo": "", "reading": "はじめる", "vocab": [("始める", "はじめる", "bắt đầu"),       ("始まり", "はじまり", "sự khởi đầu")]},

    "朝": {"viet": "TRIỀU",  "meaning_vi": "Buổi sáng, triều đình",   "meo": "", "reading": "あさ",     "vocab": [("朝ご飯", "あさごはん", "bữa sáng"),    ("今朝", "けさ", "sáng nay")]},

    "運": {"viet": "VẬN",    "meaning_vi": "Vận chuyển, vận may",     "meo": "", "reading": "はこぶ",   "vocab": [("運動", "うんどう", "vận động"),         ("運転", "うんてん", "lái xe")]},

    "終": {"viet": "CHUNG",  "meaning_vi": "Kết thúc, cuối cùng",     "meo": "", "reading": "おわる",   "vocab": [("終わる", "おわる", "kết thúc"),         ("終電", "しゅうでん", "chuyến tàu cuối")]},

    "台": {"viet": "ĐÀI",    "meaning_vi": "Bệ, đài, bộ đếm",        "meo": "", "reading": "だい",     "vocab": [("台所", "だいどころ", "nhà bếp"),        ("台風", "たいふう", "bão")]},

    "広": {"viet": "QUẢNG",  "meaning_vi": "Rộng, rộng lớn",          "meo": "", "reading": "ひろい",   "vocab": [("広い", "ひろい", "rộng"),               ("広場", "ひろば", "quảng trường")]},

    "住": {"viet": "TRÚ",    "meaning_vi": "Sống, cư trú",            "meo": "", "reading": "すむ",     "vocab": [("住む", "すむ", "sống, cư trú"),         ("住所", "じゅうしょ", "địa chỉ")]},

    "無": {"viet": "VÔ",     "meaning_vi": "Không có, vô",            "meo": "", "reading": "ない",     "vocab": [("無理", "むり", "vô lý, không thể"),     ("無料", "むりょう", "miễn phí")]},

    "真": {"viet": "CHÂN",   "meaning_vi": "Thật, sự thật",           "meo": "", "reading": "まこと",   "vocab": [("真面目", "まじめ", "nghiêm túc"),       ("真っ直ぐ", "まっすぐ", "thẳng")]},

    "有": {"viet": "HỮU",    "meaning_vi": "Có, tồn tại",             "meo": "", "reading": "ある",     "vocab": [("有名", "ゆうめい", "nổi tiếng"),        ("有料", "ゆうりょう", "có phí")]},

    "口": {"viet": "KHẨU",   "meaning_vi": "Miệng, cửa",              "meo": "", "reading": "くち",     "vocab": [("入口", "いりぐち", "lối vào"),          ("出口", "でぐち", "lối ra")]},

    "少": {"viet": "THIỂU",  "meaning_vi": "Ít, một chút",            "meo": "", "reading": "すこし",   "vocab": [("少し", "すこし", "một chút"),           ("少年", "しょうねん", "thiếu niên")]},

    "町": {"viet": "ĐINH",   "meaning_vi": "Thị trấn, phố",           "meo": "", "reading": "まち",     "vocab": [("町(まち)", "まち", "thị trấn"),        ("下町", "したまち", "phố cổ")]},

    "料": {"viet": "LIỆU",   "meaning_vi": "Phí, nguyên liệu",        "meo": "", "reading": "りょう",   "vocab": [("料理", "りょうり", "nấu ăn"),           ("材料", "ざいりょう", "nguyên liệu")]},

    "工": {"viet": "CÔNG",   "meaning_vi": "Thợ thủ công, xây dựng",  "meo": "", "reading": "こう",     "vocab": [("工場", "こうじょう", "nhà máy"),        ("工事", "こうじ", "công trình xây dựng")]},

    "建": {"viet": "KIẾN",   "meaning_vi": "Xây dựng",                "meo": "", "reading": "たてる",   "vocab": [("建物", "たてもの", "tòa nhà"),          ("建築", "けんちく", "kiến trúc")]},

    "空": {"viet": "KHÔNG",  "meaning_vi": "Bầu trời, trống rỗng",    "meo": "", "reading": "そら",     "vocab": [("空港", "くうこう", "sân bay"),          ("空気", "くうき", "không khí")]},

    "急": {"viet": "CẤP",    "meaning_vi": "Vội vàng, khẩn cấp",      "meo": "", "reading": "いそぐ",   "vocab": [("急ぐ", "いそぐ", "vội vàng"),           ("急行", "きゅうこう", "tàu nhanh")]},

    "止": {"viet": "CHỈ",    "meaning_vi": "Dừng lại",                "meo": "", "reading": "とまる",   "vocab": [("止まる", "とまる", "dừng lại"),         ("禁止", "きんし", "cấm")]},

    "送": {"viet": "TỐNG",   "meaning_vi": "Gửi, tiễn đưa",           "meo": "", "reading": "おくる",   "vocab": [("送る", "おくる", "gửi"),                ("放送", "ほうそう", "phát sóng")]},

    "切": {"viet": "THIẾT",  "meaning_vi": "Cắt, thiết yếu",          "meo": "", "reading": "きる",     "vocab": [("切る", "きる", "cắt"),                  ("大切", "たいせつ", "quan trọng")]},

    "転": {"viet": "CHUYỂN", "meaning_vi": "Chuyển, xoay vòng",       "meo": "", "reading": "まわる",   "vocab": [("運転", "うんてん", "lái xe"),           ("転ぶ", "ころぶ", "ngã")]},

    "研": {"viet": "NGHIÊN", "meaning_vi": "Nghiên cứu, mài giũa",    "meo": "", "reading": "けん",     "vocab": [("研究", "けんきゅう", "nghiên cứu"),     ("研修", "けんしゅう", "đào tạo")]},

    "足": {"viet": "TÚC",    "meaning_vi": "Chân, đủ",                "meo": "", "reading": "あし",     "vocab": [("足りる", "たりる", "đủ"),               ("遠足", "えんそく", "dã ngoại")]},

    "楽": {"viet": "LẠC",    "meaning_vi": "Vui vẻ, âm nhạc",         "meo": "", "reading": "たのしい", "vocab": [("楽しい", "たのしい", "vui"),            ("音楽", "おんがく", "âm nhạc")]},

    "起": {"viet": "KHỞI",   "meaning_vi": "Thức dậy, bắt đầu",       "meo": "", "reading": "おきる",   "vocab": [("起きる", "おきる", "thức dậy"),         ("起こす", "おこす", "đánh thức")]},

    "着": {"viet": "TRƯỚC",  "meaning_vi": "Đến nơi, mặc (quần áo)", "meo": "", "reading": "きる",     "vocab": [("着る", "きる", "mặc"),                  ("到着", "とうちゃく", "đến nơi")]},

    "店": {"viet": "ĐIẾM",   "meaning_vi": "Cửa hàng, tiệm",          "meo": "", "reading": "みせ",     "vocab": [("お店", "おみせ", "cửa hàng"),           ("店員", "てんいん", "nhân viên cửa hàng")]},

    "病": {"viet": "BỆNH",   "meaning_vi": "Bệnh tật, ốm",            "meo": "", "reading": "やむ",     "vocab": [("病気", "びょうき", "bệnh"),             ("病院", "びょういん", "bệnh viện")]},

    "質": {"viet": "CHẤT",   "meaning_vi": "Chất lượng, bản chất",    "meo": "", "reading": "しつ",     "vocab": [("質問", "しつもん", "câu hỏi"),          ("品質", "ひんしつ", "chất lượng")]},

    "待": {"viet": "ĐÃI",    "meaning_vi": "Chờ đợi",                 "meo": "", "reading": "まつ",     "vocab": [("待つ", "まつ", "chờ"),                  ("待合室", "まちあいしつ", "phòng chờ")]},

    "試": {"viet": "THỬ",    "meaning_vi": "Thử, kiểm tra",           "meo": "", "reading": "ためす",   "vocab": [("試験", "しけん", "kỳ thi"),             ("試合", "しあい", "trận đấu")]},

    "族": {"viet": "TỘC",    "meaning_vi": "Gia tộc, dân tộc",        "meo": "", "reading": "ぞく",     "vocab": [("家族", "かぞく", "gia đình"),           ("民族", "みんぞく", "dân tộc")]},

    "銀": {"viet": "NGÂN",   "meaning_vi": "Bạc (kim loại)",          "meo": "", "reading": "ぎん",     "vocab": [("銀行", "ぎんこう", "ngân hàng"),        ("銀色", "ぎんいろ", "màu bạc")]},

    "早": {"viet": "TẢO",    "meaning_vi": "Sớm, nhanh",              "meo": "", "reading": "はやい",   "vocab": [("早い", "はやい", "sớm/nhanh"),          ("早速", "さっそく", "ngay lập tức")]},

    "映": {"viet": "ẢNH",    "meaning_vi": "Chiếu phim, phản chiếu",  "meo": "", "reading": "うつる",   "vocab": [("映画", "えいが", "phim"),               ("映像", "えいぞう", "hình ảnh")]},

    "親": {"viet": "THÂN",   "meaning_vi": "Cha mẹ, thân thiết",      "meo": "", "reading": "おや",     "vocab": [("親切", "しんせつ", "tốt bụng"),         ("両親", "りょうしん", "cha mẹ")]},

    "験": {"viet": "NGHIỆM", "meaning_vi": "Kiểm tra, trải nghiệm",   "meo": "", "reading": "けん",     "vocab": [("試験", "しけん", "kỳ thi"),             ("経験", "けいけん", "kinh nghiệm")]},

    "英": {"viet": "ANH",    "meaning_vi": "Anh (quốc), tiếng Anh",   "meo": "", "reading": "えい",     "vocab": [("英語", "えいご", "tiếng Anh"),          ("英国", "えいこく", "Anh Quốc")]},

    "医": {"viet": "Y",      "meaning_vi": "Y tế, bác sĩ",            "meo": "", "reading": "い",       "vocab": [("医者", "いしゃ", "bác sĩ"),             ("医学", "いがく", "y học")]},

    "仕": {"viet": "SĨ",     "meaning_vi": "Phục vụ, làm việc",       "meo": "", "reading": "し",       "vocab": [("仕事", "しごと", "công việc"),          ("仕方", "しかた", "cách làm")]},

    "去": {"viet": "KHỨ",    "meaning_vi": "Đi, quá khứ",             "meo": "", "reading": "さる",     "vocab": [("去年", "きょねん", "năm ngoái"),        ("過去", "かこ", "quá khứ")]},

    "味": {"viet": "VỊ",     "meaning_vi": "Vị, mùi vị",              "meo": "", "reading": "あじ",     "vocab": [("意味", "いみ", "ý nghĩa"),              ("味方", "みかた", "đồng minh")]},

    "写": {"viet": "TẢ",     "meaning_vi": "Chép, chụp ảnh",          "meo": "", "reading": "うつす",   "vocab": [("写真", "しゃしん", "ảnh chụp"),         ("写す", "うつす", "chép, chụp")]},

    "字": {"viet": "TỰ",     "meaning_vi": "Chữ, ký tự",              "meo": "", "reading": "じ",       "vocab": [("文字", "もじ", "chữ, ký tự"),          ("字幕", "じまく", "phụ đề")]},

    "答": {"viet": "ĐÁP",    "meaning_vi": "Trả lời, đáp án",         "meo": "", "reading": "こたえる", "vocab": [("答える", "こたえる", "trả lời"),        ("解答", "かいとう", "đáp án")]},

    "夜": {"viet": "DẠ",     "meaning_vi": "Ban đêm, đêm",            "meo": "", "reading": "よる",     "vocab": [("夜中", "よなか", "nửa đêm"),            ("今夜", "こんや", "tối nay")]},

    "音": {"viet": "ÂM",     "meaning_vi": "Âm thanh, tiếng",         "meo": "", "reading": "おと",     "vocab": [("音楽", "おんがく", "âm nhạc"),          ("発音", "はつおん", "phát âm")]},

    "注": {"viet": "CHÚ",    "meaning_vi": "Chú ý, rót, tưới",        "meo": "", "reading": "ちゅう",   "vocab": [("注意", "ちゅうい", "chú ý"),            ("注文", "ちゅうもん", "đặt hàng")]},

    "帰": {"viet": "QUY",    "meaning_vi": "Trở về, về nhà",          "meo": "", "reading": "かえる",   "vocab": [("帰る", "かえる", "về nhà"),             ("帰国", "きこく", "về nước")]},

    "古": {"viet": "CỔ",     "meaning_vi": "Cũ, cổ xưa",              "meo": "", "reading": "ふるい",   "vocab": [("古い", "ふるい", "cũ"),                 ("古典", "こてん", "cổ điển")]},

    "歌": {"viet": "CA",     "meaning_vi": "Bài hát, ca hát",          "meo": "", "reading": "うた",     "vocab": [("歌う", "うたう", "hát"),                ("歌手", "かしゅ", "ca sĩ")]},

    "買": {"viet": "MÃI",    "meaning_vi": "Mua",                     "meo": "", "reading": "かう",     "vocab": [("買う", "かう", "mua"),                  ("買い物", "かいもの", "mua sắm")]},

    "悪": {"viet": "ÁC",     "meaning_vi": "Xấu, ác, tệ",             "meo": "", "reading": "わるい",   "vocab": [("悪い", "わるい", "xấu, tệ"),           ("悪口", "わるくち", "nói xấu")]},

    "図": {"viet": "ĐỒ",     "meaning_vi": "Bản đồ, sơ đồ",           "meo": "", "reading": "ず",       "vocab": [("地図", "ちず", "bản đồ"),               ("図書館", "としょかん", "thư viện")]},

    "週": {"viet": "TUẦN",   "meaning_vi": "Tuần lễ",                 "meo": "", "reading": "しゅう",   "vocab": [("今週", "こんしゅう", "tuần này"),       ("来週", "らいしゅう", "tuần tới")]},

    "室": {"viet": "THẤT",   "meaning_vi": "Phòng, căn phòng",        "meo": "", "reading": "しつ",     "vocab": [("教室", "きょうしつ", "phòng học"),      ("寝室", "しんしつ", "phòng ngủ")]},

    "歩": {"viet": "BỘ",     "meaning_vi": "Đi bộ, bước chân",        "meo": "", "reading": "あるく",   "vocab": [("歩く", "あるく", "đi bộ"),              ("散歩", "さんぽ", "đi dạo")]},

    "風": {"viet": "PHONG",  "meaning_vi": "Gió, phong cách",          "meo": "", "reading": "かぜ",     "vocab": [("台風", "たいふう", "bão"),              ("風邪", "かぜ", "cảm lạnh")]},

    "紙": {"viet": "CHỈ",    "meaning_vi": "Giấy",                    "meo": "", "reading": "かみ",     "vocab": [("手紙", "てがみ", "thư"),                ("紙袋", "かみぶくろ", "túi giấy")]},

    "黒": {"viet": "HẮC",    "meaning_vi": "Đen",                     "meo": "", "reading": "くろ",     "vocab": [("黒い", "くろい", "màu đen"),            ("黒板", "こくばん", "bảng đen")]},

    "花": {"viet": "HOA",    "meaning_vi": "Hoa",                     "meo": "", "reading": "はな",     "vocab": [("花火", "はなび", "pháo hoa"),           ("生け花", "いけばな", "cắm hoa")]},

    "春": {"viet": "XUÂN",   "meaning_vi": "Mùa xuân",                "meo": "", "reading": "はる",     "vocab": [("春休み", "はるやすみ", "nghỉ xuân"),    ("春分", "しゅんぶん", "xuân phân")]},

    "赤": {"viet": "XÍCH",   "meaning_vi": "Đỏ",                      "meo": "", "reading": "あかい",   "vocab": [("赤い", "あかい", "màu đỏ"),             ("赤ちゃん", "あかちゃん", "em bé")]},

    "青": {"viet": "THANH",  "meaning_vi": "Xanh (lam/lá)",           "meo": "", "reading": "あおい",   "vocab": [("青い", "あおい", "màu xanh"),           ("青春", "せいしゅん", "tuổi trẻ")]},

    "館": {"viet": "QUÁN",   "meaning_vi": "Tòa nhà lớn, quán",       "meo": "", "reading": "かん",     "vocab": [("図書館", "としょかん", "thư viện"),     ("美術館", "びじゅつかん", "bảo tàng nghệ thuật")]},

    "屋": {"viet": "ỐC",     "meaning_vi": "Mái nhà, cửa hàng",       "meo": "", "reading": "や",       "vocab": [("部屋", "へや", "căn phòng"),            ("本屋", "ほんや", "hiệu sách")]},

    "色": {"viet": "SẮC",    "meaning_vi": "Màu sắc",                 "meo": "", "reading": "いろ",     "vocab": [("色々", "いろいろ", "nhiều loại"),       ("景色", "けしき", "phong cảnh")]},

    "走": {"viet": "TẨU",    "meaning_vi": "Chạy",                    "meo": "", "reading": "はしる",   "vocab": [("走る", "はしる", "chạy"),               ("競走", "きょうそう", "cuộc đua")]},

    "秋": {"viet": "THU",    "meaning_vi": "Mùa thu",                 "meo": "", "reading": "あき",     "vocab": [("秋分", "しゅうぶん", "thu phân"),       ("秋祭り", "あきまつり", "lễ hội mùa thu")]},

    "夏": {"viet": "HẠ",     "meaning_vi": "Mùa hè",                  "meo": "", "reading": "なつ",     "vocab": [("夏休み", "なつやすみ", "nghỉ hè"),      ("夏祭り", "なつまつり", "lễ hội mùa hè")]},

    "習": {"viet": "TẬP",    "meaning_vi": "Học, luyện tập",          "meo": "", "reading": "ならう",   "vocab": [("習う", "ならう", "học"),                ("予習", "よしゅう", "học trước")]},

    "駅": {"viet": "DỊCH",   "meaning_vi": "Nhà ga, trạm",            "meo": "", "reading": "えき",     "vocab": [("駅員", "えきいん", "nhân viên ga"),     ("駅前", "えきまえ", "trước ga")]},

    "洋": {"viet": "DƯƠNG",  "meaning_vi": "Biển, phương Tây",         "meo": "", "reading": "よう",     "vocab": [("洋服", "ようふく", "quần áo Tây"),      ("洋食", "ようしょく", "món Tây")]},

    "旅": {"viet": "LỮ",     "meaning_vi": "Du lịch, hành trình",     "meo": "", "reading": "たび",     "vocab": [("旅行", "りょこう", "du lịch"),          ("旅館", "りょかん", "nhà trọ Nhật")]},

    "服": {"viet": "PHỤC",   "meaning_vi": "Quần áo, phục vụ",        "meo": "", "reading": "ふく",     "vocab": [("洋服", "ようふく", "quần áo Tây"),      ("制服", "せいふく", "đồng phục")]},

    "夕": {"viet": "TỊCH",   "meaning_vi": "Buổi chiều tối",          "meo": "", "reading": "ゆう",     "vocab": [("夕食", "ゆうしょく", "bữa tối"),        ("夕方", "ゆうがた", "buổi chiều")]},

    "借": {"viet": "TÁ",     "meaning_vi": "Mượn, thuê",              "meo": "", "reading": "かりる",   "vocab": [("借りる", "かりる", "mượn"),             ("借金", "しゃっきん", "nợ")]},

    "曜": {"viet": "DIỆU",   "meaning_vi": "Ngày trong tuần",          "meo": "", "reading": "よう",     "vocab": [("曜日", "ようび", "ngày trong tuần"),    ("日曜日", "にちようび", "Chủ nhật")]},

    "飲": {"viet": "ẨM",     "meaning_vi": "Uống",                    "meo": "", "reading": "のむ",     "vocab": [("飲む", "のむ", "uống"),                 ("飲み物", "のみもの", "đồ uống")]},

    "肉": {"viet": "NHỤC",   "meaning_vi": "Thịt",                    "meo": "", "reading": "にく",     "vocab": [("牛肉", "ぎゅうにく", "thịt bò"),        ("肉屋", "にくや", "cửa hàng thịt")]},

    "貸": {"viet": "ĐÃI",    "meaning_vi": "Cho mượn, cho thuê",      "meo": "", "reading": "かす",     "vocab": [("貸す", "かす", "cho mượn"),             ("貸し出し", "かしだし", "cho thuê")]},

    "堂": {"viet": "ĐƯỜNG",  "meaning_vi": "Hội trường, đại sảnh",    "meo": "", "reading": "どう",     "vocab": [("食堂", "しょくどう", "căn tin"),        ("講堂", "こうどう", "hội trường")]},

    "鳥": {"viet": "ĐIỂU",   "meaning_vi": "Chim, gà",                "meo": "", "reading": "とり",     "vocab": [("小鳥", "ことり", "chim nhỏ"),           ("野鳥", "やちょう", "chim hoang")]},

    "飯": {"viet": "PHẠN",   "meaning_vi": "Cơm, bữa ăn",             "meo": "", "reading": "めし",     "vocab": [("ご飯", "ごはん", "cơm, bữa ăn"),       ("夕飯", "ゆうはん", "bữa tối")]},

    "勉": {"viet": "MIỄN",   "meaning_vi": "Cố gắng, nỗ lực",         "meo": "", "reading": "べん",     "vocab": [("勉強", "べんきょう", "học tập"),        ("勤勉", "きんべん", "chăm chỉ")]},

    "冬": {"viet": "ĐÔNG",   "meaning_vi": "Mùa đông",                "meo": "", "reading": "ふゆ",     "vocab": [("冬休み", "ふゆやすみ", "nghỉ đông"),    ("冬至", "とうじ", "đông chí")]},

    "昼": {"viet": "TRÚ",    "meaning_vi": "Ban ngày, buổi trưa",     "meo": "", "reading": "ひる",     "vocab": [("昼食", "ちゅうしょく", "bữa trưa"),     ("昼間", "ひるま", "ban ngày")]},

    "茶": {"viet": "TRÀ",    "meaning_vi": "Trà",                     "meo": "", "reading": "ちゃ",     "vocab": [("お茶", "おちゃ", "trà Nhật"),           ("茶道", "さどう", "trà đạo")]},

    "弟": {"viet": "ĐỆ",     "meaning_vi": "Em trai",                 "meo": "", "reading": "おとうと", "vocab": [("兄弟", "きょうだい", "anh em"),         ("弟子", "でし", "đệ tử")]},

    "牛": {"viet": "NGƯU",   "meaning_vi": "Bò",                      "meo": "", "reading": "うし",     "vocab": [("牛肉", "ぎゅうにく", "thịt bò"),        ("牛乳", "ぎゅうにゅう", "sữa bò")]},

    "魚": {"viet": "NGƯ",    "meaning_vi": "Cá",                      "meo": "", "reading": "さかな",   "vocab": [("金魚", "きんぎょ", "cá vàng"),          ("魚屋", "さかなや", "cửa hàng cá")]},

    "兄": {"viet": "HUYNH",  "meaning_vi": "Anh trai",                "meo": "", "reading": "あに",     "vocab": [("兄弟", "きょうだい", "anh em"),         ("お兄さん", "おにいさん", "anh trai")]},

    "犬": {"viet": "KHUYỂN", "meaning_vi": "Chó",                     "meo": "", "reading": "いぬ",     "vocab": [("子犬", "こいぬ", "chó con"),            ("番犬", "ばんけん", "chó canh")]},

    "妹": {"viet": "MUỘI",   "meaning_vi": "Em gái",                  "meo": "", "reading": "いもうと", "vocab": [("姉妹", "しまい", "chị em gái"),         ("妹さん", "いもうとさん", "em gái (người khác)")]},

    "姉": {"viet": "TỶ",     "meaning_vi": "Chị gái",                 "meo": "", "reading": "あね",     "vocab": [("姉妹", "しまい", "chị em gái"),         ("お姉さん", "おねえさん", "chị gái")]},

    "漢": {"viet": "HÁN",    "meaning_vi": "Trung Quốc, chữ Hán",     "meo": "", "reading": "かん",     "vocab": [("漢字", "かんじ", "chữ Hán/Kanji"),      ("漢語", "かんご", "từ Hán")]},

    # ── Bổ sung thêm N4 ──

    "閉": {"viet": "BẾ", "meaning_vi": "Đóng", "meo": "", "reading": "しめる", "vocab": [("閉める", "しめる", "đóng"), ("閉まる", "しまる", "bị đóng")]},

    "王": {"viet": "VƯƠNG", "meaning_vi": "Vua", "meo": "", "reading": "おう", "vocab": [("国王", "こくおう", "quốc vương"), ("王様", "おうさま", "vua")]},

    "玉": {"viet": "NGỌC", "meaning_vi": "Ngọc, quả bóng", "meo": "", "reading": "たま", "vocab": [("水玉", "みずたま", "giọt nước"), ("十円玉", "じゅうえんだま", "đồng 10 yên")]},

    "谷": {"viet": "CỐC", "meaning_vi": "Thung lũng", "meo": "", "reading": "たに", "vocab": [("谷間", "たにま", "thung lũng"), ("谷川", "たにがわ", "suối")]},

    "史": {"viet": "SỬ", "meaning_vi": "Lịch sử", "meo": "", "reading": "し", "vocab": [("歴史", "れきし", "lịch sử"), ("日本史", "にほんし", "lịch sử Nhật")]},

    "治": {"viet": "TRỊ", "meaning_vi": "Cai trị, chữa bệnh", "meo": "", "reading": "なおす", "vocab": [("政治", "せいじ", "chính trị"), ("治す", "なおす", "chữa bệnh")]},

    "星": {"viet": "TINH", "meaning_vi": "Ngôi sao", "meo": "", "reading": "ほし", "vocab": [("星", "ほし", "ngôi sao"), ("火星", "かせい", "sao Hỏa")]},

    "合": {"viet": "HỢP", "meaning_vi": "Phù hợp, hợp lại", "meo": "", "reading": "あう", "vocab": [("試合", "しあい", "trận đấu"), ("間に合う", "まにあう", "kịp lúc")]},

    "皿": {"viet": "MÃNH", "meaning_vi": "Cái đĩa", "meo": "", "reading": "さら", "vocab": [("お皿", "おさら", "cái đĩa"), ("灰皿", "はいざら", "gạt tàn")]},

    "虫": {"viet": "TRÙNG", "meaning_vi": "Côn trùng, sâu bọ", "meo": "", "reading": "むし", "vocab": [("虫歯", "むしば", "sâu răng"), ("昆虫", "こんちゅう", "côn trùng")]},

    "晩": {"viet": "VÃN", "meaning_vi": "Buổi tối", "meo": "", "reading": "ばん", "vocab": [("今晩", "こんばん", "tối nay"), ("晩ご飯", "ばんごはん", "bữa tối")]},

    "遅": {"viet": "TRÌ", "meaning_vi": "Muộn, chậm", "meo": "", "reading": "おそい", "vocab": [("遅い", "おそい", "chậm, muộn"), ("遅刻", "ちこく", "đi muộn")]},

    "様": {"viet": "DẠNG", "meaning_vi": "Ngài, hình dáng", "meo": "", "reading": "さま", "vocab": [("お客様", "おきゃくさま", "quý khách"), ("様子", "ようす", "tình trạng")]},

    "雲": {"viet": "VÂN", "meaning_vi": "Đám mây", "meo": "", "reading": "くも", "vocab": [("雲", "くも", "đám mây"), ("雨雲", "あまぐも", "mây mưa")]},

    "交": {"viet": "GIAO", "meaning_vi": "Giao nhau, cắt nhau", "meo": "", "reading": "こう", "vocab": [("交通", "こうつう", "giao thông"), ("交番", "こうばん", "đồn cảnh sát")]},

    "県": {"viet": "HUYỆN", "meaning_vi": "Tỉnh", "meo": "", "reading": "けん", "vocab": [("都道府県", "とどうふけん", "các tỉnh thành"), ("県知事", "けんちじ", "tỉnh trưởng")]},

    "宿": {"viet": "TÚC", "meaning_vi": "Chỗ trọ", "meo": "", "reading": "やど", "vocab": [("宿題", "しゅくだい", "bài tập"), ("新宿", "しんじゅく", "Shinjuku")]},

    "原": {"viet": "NGUYÊN", "meaning_vi": "Thảo nguyên, nguyên bản", "meo": "", "reading": "はら", "vocab": [("原因", "げんいん", "nguyên nhân"), ("野原", "のはら", "cánh đồng")]},

    "晴": {"viet": "TÌNH", "meaning_vi": "Trời nắng, trong xanh", "meo": "", "reading": "はれる", "vocab": [("晴れ", "はれ", "trời nắng"), ("素晴らしい", "すばらしい", "tuyệt vời")]},

    "働": {"viet": "ĐỘNG", "meaning_vi": "Làm việc", "meo": "", "reading": "はたらく", "vocab": [("働く", "はたらく", "làm việc"), ("労働", "ろうどう", "lao động")]},

    "才": {"viet": "TÀI", "meaning_vi": "Tuổi, tài năng", "meo": "", "reading": "さい", "vocab": [("天才", "てんさい", "thiên tài"), ("何才", "なんさい", "mấy tuổi")]},

    "草": {"viet": "THẢO", "meaning_vi": "Cỏ", "meo": "", "reading": "くさ", "vocab": [("草", "くさ", "cỏ"), ("雑草", "ざっそう", "cỏ dại")]},

    "里": {"viet": "LÝ", "meaning_vi": "Làng quê", "meo": "", "reading": "さと", "vocab": [("古里", "ふるさと", "quê hương"), ("一里", "いちり", "một lý")]},

    "都": {"viet": "ĐÔ", "meaning_vi": "Thủ đô, đô thị", "meo": "", "reading": "と", "vocab": [("京都", "きょうと", "Kyoto"), ("都会", "とかい", "đô thị")]},

    "荷": {"viet": "HÀ", "meaning_vi": "Hành lý", "meo": "", "reading": "に", "vocab": [("荷物", "にもつ", "hành lý"), ("手荷物", "てにもつ", "hành lý xách tay")]},

    "羽": {"viet": "VŨ", "meaning_vi": "Lông chim, cánh", "meo": "", "reading": "はね", "vocab": [("羽", "はね", "lông chim"), ("千羽鶴", "せんばづる", "1000 hạc giấy")]},

    "辺": {"viet": "BIÊN", "meaning_vi": "Vùng, ven", "meo": "", "reading": "へん", "vocab": [("この辺", "このへん", "quanh đây"), ("海辺", "うみべ", "bãi biển")]},

    "糸": {"viet": "MỊCH", "meaning_vi": "Sợi chỉ", "meo": "", "reading": "いと", "vocab": [("糸", "いと", "sợi chỉ"), ("毛糸", "けいと", "sợi len")]},

    "所": {"viet": "SỞ", "meaning_vi": "Nơi chốn", "meo": "", "reading": "ところ", "vocab": [("場所", "ばしょ", "địa điểm"), ("近所", "きんじょ", "hàng xóm")]},

    "数": {"viet": "SỐ", "meaning_vi": "Số lượng, đếm", "meo": "", "reading": "かず", "vocab": [("数字", "すうじ", "chữ số"), ("数学", "すうがく", "toán học")]},

    "客": {"viet": "KHÁCH", "meaning_vi": "Khách hàng", "meo": "", "reading": "きゃく", "vocab": [("お客さん", "おきゃくさん", "khách hàng"), ("乗客", "じょうきゃく", "hành khách")]},

    "若": {"viet": "NHƯỢC", "meaning_vi": "Trẻ", "meo": "", "reading": "わかい", "vocab": [("若い", "わかい", "trẻ"), ("若者", "わかもの", "giới trẻ")]},

    "池": {"viet": "TRÌ", "meaning_vi": "Cái ao", "meo": "", "reading": "いけ", "vocab": [("池", "いけ", "cái ao"), ("電池", "でんち", "cục pin")]},

    "練": {"viet": "LUYỆN", "meaning_vi": "Luyện tập", "meo": "", "reading": "れん", "vocab": [("練習", "れんしゅう", "luyện tập"), ("訓練", "くんれん", "huấn luyện")]},

    "低": {"viet": "ĐÊ", "meaning_vi": "Thấp", "meo": "", "reading": "ひくい", "vocab": [("低い", "ひくい", "thấp"), ("最低", "さいてい", "tệ nhất")]},

    "乗": {"viet": "THỪA", "meaning_vi": "Lên xe, cưỡi", "meo": "", "reading": "のる", "vocab": [("乗る", "のる", "lên xe"), ("乗り物", "のりもの", "phương tiện")]},

    "政": {"viet": "CHÍNH", "meaning_vi": "Chính quyền", "meo": "", "reading": "せい", "vocab": [("政治", "せいじ", "chính trị"), ("政府", "せいふ", "chính phủ")]},

    "部": {"viet": "BỘ", "meaning_vi": "Bộ phận, câu lạc bộ", "meo": "", "reading": "ぶ", "vocab": [("部屋", "へや", "căn phòng"), ("部長", "ぶちょう", "trưởng phòng")]},

    "府": {"viet": "PHỦ", "meaning_vi": "Chính phủ, tỉnh", "meo": "", "reading": "ふ", "vocab": [("政府", "せいふ", "chính phủ"), ("大阪府", "おおさかふ", "tỉnh Osaka")]},

    "油": {"viet": "DU", "meaning_vi": "Dầu", "meo": "", "reading": "あぶら", "vocab": [("石油", "せきゆ", "dầu mỏ"), ("しょう油", "しょうゆ", "nước tương")]},

    "黄": {"viet": "HOÀNG", "meaning_vi": "Màu vàng", "meo": "", "reading": "き", "vocab": [("黄色", "きいろ", "màu vàng"), ("黄金", "おうごん", "vàng")]},

    "園": {"viet": "VIÊN", "meaning_vi": "Vườn", "meo": "", "reading": "えん", "vocab": [("公園", "こうえん", "công viên"), ("動物園", "どうぶつえん", "sở thú")]},

    "経": {"viet": "KINH", "meaning_vi": "Kinh tế, trải qua", "meo": "", "reading": "けい", "vocab": [("経済", "けいざい", "kinh tế"), ("経験", "けいけん", "kinh nghiệm")]},

    "区": {"viet": "KHU", "meaning_vi": "Khu vực, quận", "meo": "", "reading": "く", "vocab": [("区役所", "くやくしょ", "uỷ ban quận"), ("区別", "くべつ", "phân biệt")]},

    "短": {"viet": "ĐOẢN", "meaning_vi": "Ngắn", "meo": "", "reading": "みじかい", "vocab": [("短い", "みじかい", "ngắn"), ("短所", "たんしょ", "sở đoản")]},

    "然": {"viet": "NHIÊN", "meaning_vi": "Tự nhiên, thiên nhiên", "meo": "", "reading": "ぜん", "vocab": [("自然", "しぜん", "tự nhiên"), ("全然", "ぜんぜん", "hoàn toàn không")]},

    "馬": {"viet": "MÃ", "meaning_vi": "Con ngựa", "meo": "", "reading": "うま", "vocab": [("馬", "うま", "con ngựa"), ("乗馬", "じょうば", "cưỡi ngựa")]},

    "歴": {"viet": "LỊCH", "meaning_vi": "Lịch sử, trải qua", "meo": "", "reading": "れき", "vocab": [("歴史", "れきし", "lịch sử"), ("学歴", "がくれき", "bằng cấp")]},

    "衣": {"viet": "Y", "meaning_vi": "Áo, y phục", "meo": "", "reading": "い", "vocab": [("衣服", "いふく", "quần áo"), ("衣類", "いるい", "quần áo")]},

    "号": {"viet": "HIỆU", "meaning_vi": "Số, phiên hiệu", "meo": "", "reading": "ごう", "vocab": [("番号", "ばんごう", "số"), ("信号", "しんごう", "đèn giao thông")]},

    "声": {"viet": "THANH", "meaning_vi": "Giọng nói, âm thanh", "meo": "", "reading": "こえ", "vocab": [("声", "こえ", "giọng nói"), ("大声", "おおごえ", "tiếng lớn")]},

    "番": {"viet": "PHIÊN", "meaning_vi": "Lượt, phiên, số", "meo": "", "reading": "ばん", "vocab": [("一番", "いちばん", "số một"), ("番組", "ばんぐみ", "chương trình")]},

    "妻": {"viet": "THÊ", "meaning_vi": "Vợ", "meo": "", "reading": "つま", "vocab": [("妻", "つま", "vợ (mình)"), ("夫婦", "ふうふ", "vợ chồng")]},

    "船": {"viet": "THUYỀN", "meaning_vi": "Con thuyền", "meo": "", "reading": "ふね", "vocab": [("船", "ふね", "thuyền"), ("船便", "ふなびん", "đường thủy")]},

    "緑": {"viet": "LỤC", "meaning_vi": "Màu xanh lá", "meo": "", "reading": "みどり", "vocab": [("緑", "みどり", "màu xanh lá"), ("緑茶", "りょくちゃ", "trà xanh")]},

    "宅": {"viet": "TRẠCH", "meaning_vi": "Nhà", "meo": "", "reading": "たく", "vocab": [("自宅", "じたく", "nhà mình"), ("お宅", "おたく", "nhà người khác")]},

    "済": {"viet": "TẾ", "meaning_vi": "Hoàn tất, kinh tế", "meo": "", "reading": "すむ", "vocab": [("経済", "けいざい", "kinh tế"), ("済む", "すむ", "hoàn tất")]},

    "鉄": {"viet": "THIẾT", "meaning_vi": "Sắt, thép", "meo": "", "reading": "てつ", "vocab": [("地下鉄", "ちかてつ", "tàu điện ngầm"), ("鉄道", "てつどう", "đường sắt")]},

    "育": {"viet": "DỤC", "meaning_vi": "Giáo dục, nuôi dưỡng", "meo": "", "reading": "そだてる", "vocab": [("育てる", "そだてる", "nuôi dưỡng"), ("教育", "きょういく", "giáo dục")]},

    "鳴": {"viet": "MINH", "meaning_vi": "Hót, kêu", "meo": "", "reading": "なく", "vocab": [("鳴く", "なく", "kêu, hót"), ("鳴る", "なる", "reo, kêu")]},

}



# ── N4 bổ sung từ MNN Book 2 (chưa có trong N4_VI) ──────────────────────────

MNN_N4_EXTRA: dict[str, dict] = {

    "父": {"viet": "PHỤ",    "meaning_vi": "Cha, bố",                 "meo": "", "reading": "ちち",     "vocab": [("父親", "ちちおや", "người cha"),         ("お父さん", "おとうさん", "bố (lịch sự)")]},

    "母": {"viet": "MẪU",    "meaning_vi": "Mẹ",                      "meo": "", "reading": "はは",     "vocab": [("母親", "ははおや", "người mẹ"),          ("お母さん", "おかあさん", "mẹ (lịch sự)")]},

    "友": {"viet": "HỮU",    "meaning_vi": "Bạn bè",                  "meo": "", "reading": "とも",     "vocab": [("友達", "ともだち", "bạn bè"),            ("友人", "ゆうじん", "bạn bè (trang trọng)")]},

    "男": {"viet": "NAM",    "meaning_vi": "Nam, đàn ông",             "meo": "", "reading": "おとこ",   "vocab": [("男の人", "おとこのひと", "đàn ông"),     ("男性", "だんせい", "nam giới")]},

    "女": {"viet": "NỮ",     "meaning_vi": "Nữ, phụ nữ",              "meo": "", "reading": "おんな",   "vocab": [("女の人", "おんなのひと", "phụ nữ"),      ("女性", "じょせい", "nữ giới")]},

    "子": {"viet": "TỬ",     "meaning_vi": "Con, đứa trẻ",            "meo": "", "reading": "こ",       "vocab": [("子供", "こども", "trẻ em"),              ("女の子", "おんなのこ", "bé gái")]},

    "白": {"viet": "BẠCH",   "meaning_vi": "Trắng",                   "meo": "", "reading": "しろ",     "vocab": [("白い", "しろい", "màu trắng"),           ("白黒", "しろくろ", "đen trắng")]},

    "長": {"viet": "TRƯỜNG", "meaning_vi": "Dài, trưởng",             "meo": "", "reading": "ながい",   "vocab": [("長い", "ながい", "dài"),                 ("社長", "しゃちょう", "giám đốc")]},

    "遠": {"viet": "VIỄN",   "meaning_vi": "Xa",                      "meo": "", "reading": "とおい",   "vocab": [("遠い", "とおい", "xa"),                  ("遠足", "えんそく", "dã ngoại")]},

    "高": {"viet": "CAO",    "meaning_vi": "Cao, đắt",                 "meo": "", "reading": "たかい",   "vocab": [("高い", "たかい", "cao/đắt"),             ("最高", "さいこう", "tuyệt vời nhất")]},

    "村": {"viet": "THÔN",   "meaning_vi": "Làng, thôn",              "meo": "", "reading": "むら",     "vocab": [("田舎", "いなか", "nông thôn"),           ("村人", "むらびと", "dân làng")]},

    "市": {"viet": "THỊ",    "meaning_vi": "Thành phố, thị trường",   "meo": "", "reading": "し",       "vocab": [("市場", "いちば", "chợ"),                 ("都市", "とし", "thành phố")]},

    "天": {"viet": "THIÊN",  "meaning_vi": "Trời, bầu trời",          "meo": "", "reading": "てん",     "vocab": [("天気", "てんき", "thời tiết"),           ("天国", "てんごく", "thiên đường")]},

    "気": {"viet": "KHÍ",    "meaning_vi": "Khí, cảm giác, thời tiết","meo": "", "reading": "き",       "vocab": [("天気", "てんき", "thời tiết"),           ("元気", "げんき", "khỏe mạnh")]},

    "雨": {"viet": "VŨ",     "meaning_vi": "Mưa",                     "meo": "", "reading": "あめ",     "vocab": [("雨降り", "あめふり", "trời mưa"),        ("大雨", "おおあめ", "mưa to")]},

    "雪": {"viet": "TUYẾT",  "meaning_vi": "Tuyết",                   "meo": "", "reading": "ゆき",     "vocab": [("雪国", "ゆきぐに", "xứ tuyết"),          ("大雪", "おおゆき", "tuyết lớn")]},

    "暑": {"viet": "THỬ",    "meaning_vi": "Nóng (thời tiết)",        "meo": "", "reading": "あつい",   "vocab": [("暑い", "あつい", "nóng"),                ("猛暑", "もうしょ", "nắng nóng gay gắt")]},

    "寒": {"viet": "HÀN",    "meaning_vi": "Lạnh (thời tiết)",        "meo": "", "reading": "さむい",   "vocab": [("寒い", "さむい", "lạnh"),                ("寒波", "かんぱ", "đợt lạnh")]},

    "暖": {"viet": "NOÃN",   "meaning_vi": "Ấm áp",                   "meo": "", "reading": "あたたかい","vocab": [("暖かい", "あたたかい", "ấm áp"),        ("暖房", "だんぼう", "máy sưởi")]},

    "涼": {"viet": "LƯƠNG",  "meaning_vi": "Mát mẻ",                  "meo": "", "reading": "すずしい", "vocab": [("涼しい", "すずしい", "mát mẻ"),         ("涼風", "すずかぜ", "gió mát")]},

    "暗": {"viet": "ÁM",     "meaning_vi": "Tối, tăm tối",            "meo": "", "reading": "くらい",   "vocab": [("暗い", "くらい", "tối"),                 ("暗記", "あんき", "học thuộc lòng")]},

    "静": {"viet": "TĨNH",   "meaning_vi": "Yên tĩnh",                "meo": "", "reading": "しずかな", "vocab": [("静かな", "しずかな", "yên tĩnh"),       ("静止", "せいし", "đứng yên")]},

    "太": {"viet": "THÁI",   "meaning_vi": "Béo, to lớn",             "meo": "", "reading": "ふとい",   "vocab": [("太い", "ふとい", "to/béo"),              ("太平洋", "たいへいよう", "Thái Bình Dương")]},

    "細": {"viet": "TẾ",     "meaning_vi": "Mỏng, nhỏ, chi tiết",    "meo": "", "reading": "ほそい",   "vocab": [("細い", "ほそい", "mảnh/nhỏ"),           ("詳細", "しょうさい", "chi tiết")]},

    "弱": {"viet": "NHƯỢC",  "meaning_vi": "Yếu",                     "meo": "", "reading": "よわい",   "vocab": [("弱い", "よわい", "yếu"),                ("弱点", "じゃくてん", "điểm yếu")]},

    "軽": {"viet": "KHINH",  "meaning_vi": "Nhẹ",                     "meo": "", "reading": "かるい",   "vocab": [("軽い", "かるい", "nhẹ"),                ("軽食", "けいしょく", "ăn nhẹ")]},

    "速": {"viet": "TỐC",    "meaning_vi": "Nhanh, tốc độ",           "meo": "", "reading": "はやい",   "vocab": [("速い", "はやい", "nhanh"),               ("速度", "そくど", "tốc độ")]},

    "便": {"viet": "TIỆN",   "meaning_vi": "Tiện lợi, bưu kiện",      "meo": "", "reading": "べん",     "vocab": [("便利", "べんり", "tiện lợi"),            ("不便", "ふべん", "bất tiện")]},

    "利": {"viet": "LỢI",    "meaning_vi": "Lợi ích, thuận tiện",     "meo": "", "reading": "り",       "vocab": [("便利", "べんり", "tiện lợi"),            ("利用", "りよう", "sử dụng")]},

    "薬": {"viet": "DƯỢC",   "meaning_vi": "Thuốc",                   "meo": "", "reading": "くすり",   "vocab": [("薬局", "やっきょく", "nhà thuốc"),       ("飲み薬", "のみぐすり", "thuốc uống")]},

    "丈": {"viet": "TRƯỢNG", "meaning_vi": "Cao, khỏe mạnh",          "meo": "", "reading": "じょう",   "vocab": [("大丈夫", "だいじょうぶ", "ổn thôi"),    ("丈夫", "じょうぶ", "chắc chắn/khỏe")]},

    "夫": {"viet": "PHU",    "meaning_vi": "Chồng, người đàn ông",    "meo": "", "reading": "おっと",   "vocab": [("夫婦", "ふうふ", "vợ chồng"),           ("大丈夫", "だいじょうぶ", "ổn thôi")]},

    "配": {"viet": "PHỐI",   "meaning_vi": "Phân phối, lo lắng",      "meo": "", "reading": "くばる",   "vocab": [("心配", "しんぱい", "lo lắng"),          ("配達", "はいたつ", "giao hàng")]},

    # ── Bổ sung thêm ────────────────────────────────────────────────────────

    "麦": {"viet": "MẠCH",   "meaning_vi": "Lúa mạch, lúa mì",       "meo": "", "reading": "むぎ",     "vocab": [("小麦", "こむぎ", "lúa mì"),             ("麦茶", "むぎちゃ", "trà lúa mạch")]},

    "化": {"viet": "HÓA",    "meaning_vi": "Biến hóa, hóa học",       "meo": "", "reading": "か",       "vocab": [("文化", "ぶんか", "văn hóa"),            ("変化", "へんか", "thay đổi")]},

    "奥": {"viet": "ÁO",     "meaning_vi": "Sâu bên trong, vợ",       "meo": "", "reading": "おく",     "vocab": [("奥さん", "おくさん", "vợ (lịch sự)"),  ("奥深い", "おくふかい", "sâu sắc")]},

    "科": {"viet": "KHOA",   "meaning_vi": "Khoa, môn học, chuyên ngành","meo": "","reading": "か",     "vocab": [("科学", "かがく", "khoa học"),            ("教科書", "きょうかしょ", "sách giáo khoa")]},

    "世": {"viet": "THẾ",    "meaning_vi": "Thế giới, đời",           "meo": "", "reading": "せ",       "vocab": [("世界", "せかい", "thế giới"),           ("世紀", "せいき", "thế kỷ")]},

    "有": {"viet": "HỮU",    "meaning_vi": "Có, sở hữu",              "meo": "", "reading": "ある",     "vocab": [("有名", "ゆうめい", "nổi tiếng"),        ("有効", "ゆうこう", "có hiệu lực")]},


    "土": {"viet": "土", "meaning_vi": "", "reading": "", "vocab": []},
    "大": {"viet": "大", "meaning_vi": "", "reading": "", "vocab": []},
    "小": {"viet": "小", "meaning_vi": "", "reading": "", "vocab": []},
    "木": {"viet": "木", "meaning_vi": "", "reading": "", "vocab": []},
    "水": {"viet": "水", "meaning_vi": "", "reading": "", "vocab": []},
    "火": {"viet": "火", "meaning_vi": "", "reading": "", "vocab": []},
    "金": {"viet": "金", "meaning_vi": "", "reading": "", "vocab": []},
}



# ── N5 kanji từ MNN Book 1 ────────────────────────────────────────────────────

MNN_N5: dict[str, dict] = {

    "一": {"viet": "NHẤT",   "meaning_vi": "Một",                     "meo": "", "reading": "いち",     "vocab": [("一番", "いちばん", "số một/nhất"),       ("一緒", "いっしょ", "cùng nhau")]},

    "二": {"viet": "NHỊ",    "meaning_vi": "Hai",                     "meo": "", "reading": "に",       "vocab": [("二人", "ふたり", "hai người"),           ("二番", "にばん", "thứ hai")]},

    "三": {"viet": "TAM",    "meaning_vi": "Ba",                      "meo": "", "reading": "さん",     "vocab": [("三人", "さんにん", "ba người"),          ("三角", "さんかく", "tam giác")]},

    "四": {"viet": "TỨ",     "meaning_vi": "Bốn",                     "meo": "", "reading": "よん",     "vocab": [("四人", "よにん", "bốn người"),           ("四角", "しかく", "hình vuông")]},

    "五": {"viet": "NGŨ",    "meaning_vi": "Năm",                     "meo": "", "reading": "ご",       "vocab": [("五人", "ごにん", "năm người"),           ("五月", "ごがつ", "tháng Năm")]},

    "六": {"viet": "LỤC",    "meaning_vi": "Sáu",                     "meo": "", "reading": "ろく",     "vocab": [("六人", "ろくにん", "sáu người"),         ("六月", "ろくがつ", "tháng Sáu")]},

    "七": {"viet": "THẤT",   "meaning_vi": "Bảy",                     "meo": "", "reading": "なな",     "vocab": [("七人", "しちにん", "bảy người"),         ("七月", "しちがつ", "tháng Bảy")]},

    "八": {"viet": "BÁT",    "meaning_vi": "Tám",                     "meo": "", "reading": "はち",     "vocab": [("八人", "はちにん", "tám người"),         ("八月", "はちがつ", "tháng Tám")]},

    "九": {"viet": "CỬU",    "meaning_vi": "Chín",                    "meo": "", "reading": "きゅう",   "vocab": [("九人", "くにん", "chín người"),          ("九月", "くがつ", "tháng Chín")]},

    "十": {"viet": "THẬP",   "meaning_vi": "Mười",                    "meo": "", "reading": "じゅう",   "vocab": [("十人", "じゅうにん", "mười người"),      ("十分", "じゅうぶん", "đủ rồi")]},

    "百": {"viet": "BÁCH",   "meaning_vi": "Một trăm",                "meo": "", "reading": "ひゃく",   "vocab": [("百円", "ひゃくえん", "100 yên"),         ("百科", "ひゃっか", "bách khoa")]},

    "千": {"viet": "THIÊN",  "meaning_vi": "Một nghìn",               "meo": "", "reading": "せん",     "vocab": [("千円", "せんえん", "1000 yên"),          ("千年", "せんねん", "nghìn năm")]},

    "万": {"viet": "VẠN",    "meaning_vi": "Mười nghìn",              "meo": "", "reading": "まん",     "vocab": [("一万円", "いちまんえん", "10.000 yên"),  ("万年筆", "まんねんひつ", "bút máy")]},

    "円": {"viet": "VIÊN",   "meaning_vi": "Yên (tiền), tròn",        "meo": "", "reading": "えん",     "vocab": [("円", "えん", "yên Nhật"),               ("円い", "まるい", "tròn")]},

    "年": {"viet": "NIÊN",   "meaning_vi": "Năm (thời gian)",         "meo": "", "reading": "ねん",     "vocab": [("今年", "ことし", "năm nay"),             ("来年", "らいねん", "năm tới")]},

    "月": {"viet": "NGUYỆT", "meaning_vi": "Tháng, mặt trăng",       "meo": "", "reading": "つき",     "vocab": [("今月", "こんげつ", "tháng này"),         ("来月", "らいげつ", "tháng tới")]},

    "日": {"viet": "NHẬT",   "meaning_vi": "Ngày, mặt trời, Nhật",   "meo": "", "reading": "ひ",       "vocab": [("今日", "きょう", "hôm nay"),             ("誕生日", "たんじょうび", "sinh nhật")]},

    "時": {"viet": "THỜI",   "meaning_vi": "Giờ, thời gian",          "meo": "", "reading": "じ",       "vocab": [("何時", "なんじ", "mấy giờ"),             ("時間", "じかん", "thời gian")]},

    "分": {"viet": "PHÂN",   "meaning_vi": "Phút, chia",              "meo": "", "reading": "ふん",     "vocab": [("何分", "なんぷん", "mấy phút"),          ("自分", "じぶん", "bản thân")]},

    "上": {"viet": "THƯỢNG", "meaning_vi": "Trên, lên",               "meo": "", "reading": "うえ",     "vocab": [("上手", "じょうず", "giỏi"),              ("屋上", "おくじょう", "sân thượng")]},

    "下": {"viet": "HẠ",     "meaning_vi": "Dưới, xuống",             "meo": "", "reading": "した",     "vocab": [("地下", "ちか", "tầng hầm"),             ("下手", "へた", "kém, vụng")]},

    "外": {"viet": "NGOẠI",  "meaning_vi": "Ngoài, bên ngoài",        "meo": "", "reading": "そと",     "vocab": [("外国", "がいこく", "nước ngoài"),        ("外出", "がいしゅつ", "ra ngoài")]},

    "右": {"viet": "HỮU",    "meaning_vi": "Bên phải",                "meo": "", "reading": "みぎ",     "vocab": [("右手", "みぎて", "tay phải"),            ("右折", "うせつ", "rẽ phải")]},

    "左": {"viet": "TẢ",     "meaning_vi": "Bên trái",                "meo": "", "reading": "ひだり",   "vocab": [("左手", "ひだりて", "tay trái"),          ("左折", "させつ", "rẽ trái")]},

    "前": {"viet": "TIỀN",   "meaning_vi": "Trước",                   "meo": "", "reading": "まえ",     "vocab": [("名前", "なまえ", "tên"),                 ("前日", "ぜんじつ", "ngày hôm trước")]},

    "後": {"viet": "HẬU",    "meaning_vi": "Sau, phía sau",           "meo": "", "reading": "あと",     "vocab": [("午後", "ごご", "buổi chiều/PM"),         ("後ろ", "うしろ", "phía sau")]},

    "東": {"viet": "ĐÔNG",   "meaning_vi": "Phía Đông",               "meo": "", "reading": "ひがし",   "vocab": [("東京", "とうきょう", "Tokyo"),           ("東口", "ひがしぐち", "cửa phía Đông")]},

    "西": {"viet": "TÂY",    "meaning_vi": "Phía Tây",                "meo": "", "reading": "にし",     "vocab": [("関西", "かんさい", "vùng Kansai"),       ("西口", "にしぐち", "cửa phía Tây")]},

    "南": {"viet": "NAM",    "meaning_vi": "Phía Nam",                "meo": "", "reading": "みなみ",   "vocab": [("南口", "みなみぐち", "cửa phía Nam"),    ("東南", "とうなん", "Đông Nam")]},

    "北": {"viet": "BẮC",    "meaning_vi": "Phía Bắc",                "meo": "", "reading": "きた",     "vocab": [("北口", "きたぐち", "cửa phía Bắc"),     ("北海道", "ほっかいどう", "Hokkaido")]},

    "人": {"viet": "NHÂN",   "meaning_vi": "Người",                   "meo": "", "reading": "ひと",     "vocab": [("外国人", "がいこくじん", "người nước ngoài"),("日本人", "にほんじん", "người Nhật")]},

    "本": {"viet": "BẢN",    "meaning_vi": "Sách, gốc, Nhật Bản",    "meo": "", "reading": "ほん",     "vocab": [("日本", "にほん", "Nhật Bản"),            ("本屋", "ほんや", "hiệu sách")]},

    "国": {"viet": "QUỐC",   "meaning_vi": "Đất nước, quốc gia",      "meo": "", "reading": "くに",     "vocab": [("外国", "がいこく", "nước ngoài"),        ("国際", "こくさい", "quốc tế")]},

    "語": {"viet": "NGỮ",    "meaning_vi": "Ngôn ngữ, lời nói",       "meo": "", "reading": "ご",       "vocab": [("日本語", "にほんご", "tiếng Nhật"),      ("英語", "えいご", "tiếng Anh")]},

    "電": {"viet": "ĐIỆN",   "meaning_vi": "Điện",                    "meo": "", "reading": "でん",     "vocab": [("電車", "でんしゃ", "tàu điện"),          ("電話", "でんわ", "điện thoại")]},

    "車": {"viet": "XA",     "meaning_vi": "Xe",                      "meo": "", "reading": "くるま",   "vocab": [("電車", "でんしゃ", "tàu điện"),          ("自転車", "じてんしゃ", "xe đạp")]},

    "駅": {"viet": "DỊCH",   "meaning_vi": "Nhà ga",                  "meo": "", "reading": "えき",     "vocab": [("駅員", "えきいん", "nhân viên ga"),      ("駅前", "えきまえ", "trước ga")]},

    "学": {"viet": "HỌC",    "meaning_vi": "Học, khoa học",           "meo": "", "reading": "がく",     "vocab": [("大学", "だいがく", "đại học"),           ("学生", "がくせい", "sinh viên")]},

    "校": {"viet": "HIỆU",   "meaning_vi": "Trường học",              "meo": "", "reading": "こう",     "vocab": [("学校", "がっこう", "trường học"),        ("高校", "こうこう", "trường THPT")]},

    "先": {"viet": "TIÊN",   "meaning_vi": "Trước, thầy, đầu tiên",  "meo": "", "reading": "さき",     "vocab": [("先生", "せんせい", "thầy/cô giáo"),     ("先週", "せんしゅう", "tuần trước")]},

    "生": {"viet": "SINH",   "meaning_vi": "Sinh ra, học sinh",       "meo": "", "reading": "せい",     "vocab": [("学生", "がくせい", "sinh viên"),         ("先生", "せんせい", "giáo viên")]},

    "今": {"viet": "KIM",    "meaning_vi": "Bây giờ, hiện tại",       "meo": "", "reading": "いま",     "vocab": [("今日", "きょう", "hôm nay"),             ("今週", "こんしゅう", "tuần này")]},

    "何": {"viet": "HÀ",     "meaning_vi": "Cái gì, bao nhiêu",       "meo": "", "reading": "なに",     "vocab": [("何時", "なんじ", "mấy giờ"),             ("何人", "なんにん", "bao nhiêu người")]},

    "名": {"viet": "DANH",   "meaning_vi": "Tên, danh tiếng",         "meo": "", "reading": "な",       "vocab": [("名前", "なまえ", "tên"),                 ("有名", "ゆうめい", "nổi tiếng")]},

    "入": {"viet": "NHẬP",   "meaning_vi": "Vào, nhập",               "meo": "", "reading": "はいる",   "vocab": [("入口", "いりぐち", "lối vào"),           ("入学", "にゅうがく", "nhập học")]},

    "食": {"viet": "THỰC",   "meaning_vi": "Ăn, thức ăn",             "meo": "", "reading": "たべる",   "vocab": [("食べ物", "たべもの", "đồ ăn"),           ("食堂", "しょくどう", "căn tin")]},

    "飲": {"viet": "ẨM",     "meaning_vi": "Uống",                    "meo": "", "reading": "のむ",     "vocab": [("飲み物", "のみもの", "đồ uống"),         ("飲食", "いんしょく", "ăn uống")]},

    "見": {"viet": "KIẾN",   "meaning_vi": "Nhìn, thấy",              "meo": "", "reading": "みる",     "vocab": [("見る", "みる", "xem, nhìn"),             ("見学", "けんがく", "tham quan")]},

    "行": {"viet": "HÀNH",   "meaning_vi": "Đi, hành động",           "meo": "", "reading": "いく",     "vocab": [("旅行", "りょこう", "du lịch"),           ("銀行", "ぎんこう", "ngân hàng")]},

    "来": {"viet": "LAI",    "meaning_vi": "Đến, lai",                "meo": "", "reading": "くる",     "vocab": [("来る", "くる", "đến"),                  ("来週", "らいしゅう", "tuần tới")]},

    "帰": {"viet": "QUY",    "meaning_vi": "Về nhà, trở về",          "meo": "", "reading": "かえる",   "vocab": [("帰る", "かえる", "về nhà"),              ("帰国", "きこく", "về nước")]},

    "買": {"viet": "MÃI",    "meaning_vi": "Mua",                     "meo": "", "reading": "かう",     "vocab": [("買い物", "かいもの", "mua sắm"),         ("買う", "かう", "mua")]},

    "起": {"viet": "KHỞI",   "meaning_vi": "Dậy, bắt đầu",           "meo": "", "reading": "おきる",   "vocab": [("起きる", "おきる", "thức dậy"),          ("起こす", "おこす", "đánh thức")]},

    "寝": {"viet": "THẨM",   "meaning_vi": "Ngủ, đi ngủ",            "meo": "", "reading": "ねる",     "vocab": [("寝る", "ねる", "đi ngủ"),                ("寝室", "しんしつ", "phòng ngủ")]},

    "休": {"viet": "HƯU",    "meaning_vi": "Nghỉ ngơi",               "meo": "", "reading": "やすむ",   "vocab": [("休む", "やすむ", "nghỉ"),                ("夏休み", "なつやすみ", "nghỉ hè")]},

    "書": {"viet": "THƯ",    "meaning_vi": "Viết, sách",              "meo": "", "reading": "かく",     "vocab": [("書く", "かく", "viết"),                  ("辞書", "じしょ", "từ điển")]},

    "読": {"viet": "ĐỌC",    "meaning_vi": "Đọc",                     "meo": "", "reading": "よむ",     "vocab": [("読む", "よむ", "đọc"),                   ("読書", "どくしょ", "đọc sách")]},

    "聞": {"viet": "VĂN",    "meaning_vi": "Nghe, hỏi",               "meo": "", "reading": "きく",     "vocab": [("聞く", "きく", "nghe/hỏi"),              ("新聞", "しんぶん", "báo")]},

    "話": {"viet": "THOẠI",  "meaning_vi": "Nói chuyện, câu chuyện",  "meo": "", "reading": "はなす",   "vocab": [("話す", "はなす", "nói"),                 ("電話", "でんわ", "điện thoại")]},

    "山": {"viet": "SƠN",    "meaning_vi": "Núi",                     "meo": "", "reading": "やま",     "vocab": [("富士山", "ふじさん", "núi Phú Sĩ"),     ("山登り", "やまのぼり", "leo núi")]},

    "川": {"viet": "XUYÊN",  "meaning_vi": "Sông",                    "meo": "", "reading": "かわ",     "vocab": [("川岸", "かわぎし", "bờ sông"),           ("河川", "かせん", "sông ngòi")]},

    "田": {"viet": "ĐIỀN",   "meaning_vi": "Ruộng lúa",               "meo": "", "reading": "た",       "vocab": [("田んぼ", "たんぼ", "ruộng lúa"),        ("田舎", "いなか", "nông thôn")]},

    "午": {"viet": "NGỌ",    "meaning_vi": "12 giờ trưa",             "meo": "", "reading": "ご",       "vocab": [("午前", "ごぜん", "buổi sáng/AM"),        ("午後", "ごご", "buổi chiều/PM")]},

    "半": {"viet": "BÁN",    "meaning_vi": "Nửa",                     "meo": "", "reading": "はん",     "vocab": [("一時半", "いちじはん", "1 giờ rưỡi"),   ("半分", "はんぶん", "một nửa")]},

    "毎": {"viet": "MỖI",    "meaning_vi": "Mỗi, hàng",               "meo": "", "reading": "まい",     "vocab": [("毎日", "まいにち", "mỗi ngày"),          ("毎週", "まいしゅう", "mỗi tuần")]},

    "間": {"viet": "GIAN",   "meaning_vi": "Giữa, khoảng thời gian",  "meo": "", "reading": "あいだ",   "vocab": [("時間", "じかん", "thời gian"),           ("間に合う", "まにあう", "kịp giờ")]},

    "作": {"viet": "TÁC",    "meaning_vi": "Làm, tạo ra",             "meo": "", "reading": "つくる",   "vocab": [("作る", "つくる", "làm/nấu"),             ("作品", "さくひん", "tác phẩm")]},

    "使": {"viet": "SỬ",     "meaning_vi": "Sử dụng",                 "meo": "", "reading": "つかう",   "vocab": [("使う", "つかう", "dùng"),                ("大使館", "たいしかん", "đại sứ quán")]},

}




# ── N3 Kanji (375 chữ) ──────────────────────────────────────────────────────
N3_VI: dict[str, dict] = {
    "回": {"viet": "HỒI, HỐI", "meaning_vi": "Về, đi rồi trở lại gọi là hồi.", "meo": "", "reading": "まわ.る -まわ.る -まわ.り まわ.す -まわ.す まわ.し- -まわ.し もとお.る か.える", "vocab": [("回", "かい", "lần"), ("回す", "まわす", "quây")]},
    "向": {"viet": "HƯỚNG", "meaning_vi": "Ngoảnh về, hướng về. Ngoảnh về phương vị nào gọi là hướng. Như nam hướng [南向] ngoảnh về hướng nam, bắc hướng [北向] ngoảnh về hướng bắc, v.v. Ý chí ngả về mặt nào gọi là chí hướng [志向], xu hướng [趨向], v.v.", "meo": "", "reading": "む.く む.い -む.き む.ける -む.け む.かう む.かい む.こう む.こう- むこ むか.い", "vocab": [("向い", "むかい", "sự đương đầu"), ("向う", "むこう", "mặt")]},
    "匹": {"viet": "THẤT", "meaning_vi": "Xếp, con. Tính số vải lụa gọi là thất, đời xưa tính dài bốn trượng là một thất. Một con ngựa cũng gọi là nhất thất [一匹]. Tục cũng dùng cả chữ thất [疋].", "meo": "", "reading": "ひき", "vocab": [("匹", "ひき", "con"), ("匹", "ひつ", "con")]},
    "面": {"viet": "DIỆN, MIẾN", "meaning_vi": "Mặt, là cái bộ phận gồm cả tai, mắt, miệng, mũi.", "meo": "", "reading": "おも おもて つら", "vocab": [("面", "おも", "mặt; bề ngoài"), ("面", "つら", "bề mặt; mặt")]},
    "失": {"viet": "THẤT", "meaning_vi": "Mất. Như tam sao thất bản  [三抄失本] ba lần chép lại thì đã làm mất hết cả gốc, ý nói mỗi lần chép lại là mỗi lần sai đi.", "meo": "", "reading": "うしな.う う.せる", "vocab": [("失う", "うしなう", "mất; đánh mất; bị tước; lỡ; bỏ lỡ; bị mất; bị cướp"), ("亡失", "ぼうしつ", "sự mất")]},
    "和": {"viet": "HÒA, HỌA", "meaning_vi": "Hòa, cùng ăn nhịp với nhau.", "meo": "", "reading": "やわ.らぐ やわ.らげる なご.む なご.やか", "vocab": [("和", "わ", "hòa bình"), ("和む", "なごむ", "bình tĩnh; điềm tĩnh; nguôi đi; thư thái")]},
    "厚": {"viet": "HẬU", "meaning_vi": "Chiều dày.", "meo": "", "reading": "あつ.い あか", "vocab": [("厚い", "あつい", "dày"), ("厚さ", "あつさ", "bề dày")]},
    "存": {"viet": "TỒN", "meaning_vi": "Còn, trái lại với chữ vong [亡] mất, cho nên sinh tử cũng gọi là tồn vong [存亡].", "meo": "", "reading": "ソン ゾン", "vocab": [("並存", "へいそん", "sự chung sống"), ("存亡", "そんぼう", "tồn vong .")]},
    "在": {"viet": "TẠI", "meaning_vi": "Ở. Như tại hạ vị nhi bất ưu [在下位而不憂] ở ngôi dưới mà chẳng lo.", "meo": "", "reading": "あ.る", "vocab": [("在る", "ある", "có"), ("不在", "ふざい", "khiếm khuyết")]},
    "肯": {"viet": "KHẲNG, KHẢI", "meaning_vi": "Khá, ừ được. Bằng lòng cho gọi là khẳng. Như khẳng định [肯定] nhận là như có vậy, thừa nhận.", "meo": "", "reading": "がえんじ.る", "vocab": [("肯く", "うなずく", "cái gật đầu; sự cúi đầu ; sự ra hiệu"), ("肯定", "こうてい", "sự khẳng định")]},
    "信": {"viet": "TÍN", "meaning_vi": "Tin, không sai lời hẹn là tín. Như trung tín [忠信] tin thực.", "meo": "", "reading": "シン", "vocab": [("信", "しん", "lòng trung thành"), ("信", "まこと", "sự thật lòng; lòng chân thật .")]},
    "許": {"viet": "HỨA, HỬ, HỔ", "meaning_vi": "Nghe theo, ừ cho. Như hứa khả [許可] ừ cho là được.", "meo": "", "reading": "ゆる.す もと", "vocab": [("許し", "ゆるし", "sự tha thứ"), ("許す", "もとす", "tha lỗi")]},
    "件": {"viet": "KIỆN", "meaning_vi": "Món đồ. Tục gọi một món đồ đựng trong một cái bồ hay cái sọt là một kiện [件]. Như bưu kiện [郵件] đồ vật gửi theo đường bưu điện.", "meo": "", "reading": "くだん", "vocab": [("件", "けん", "vụ; trường hợp; vấn đề; việc"), ("一件", "いちけん", "chất")]},
    "念": {"viet": "NIỆM", "meaning_vi": "Nghĩ nhớ.", "meo": "", "reading": "ネン", "vocab": [("念", "ねん", "sự chú ý"), ("念々", "ねんねん", "sự nghĩ ngợi liên tục (về cái gì đó) .")]},
    "冷": {"viet": "LÃNH", "meaning_vi": "Lạnh.", "meo": "", "reading": "つめ.たい ひ.える ひ.や ひ.ややか ひ.やす ひ.やかす さ.める さ.ます", "vocab": [("冷や", "ひや", "nước lạnh ."), ("冷光", "ひやこう", "sự phát sáng")]},
    "介": {"viet": "GIỚI", "meaning_vi": "Cõi, ở vào khoảng giữa hai cái gọi là giới. Ngày xưa giao tiếp với nhau, chủ có người thấn mà khách có người giới [介] để giúp lễ và đem nhời người bên này nói với người bên kia biết. Như một người ở giữa nói cho người thứ nhất và người thứ ba biết nhau mà làm quen nhau gọi là giới thiệu [介紹] hay môi giới [媒介] v.v.", "meo": "", "reading": "カイ", "vocab": [("介", "かい", "vỏ; bao; mai"), ("一介", "いっかい", "ao; hồ")]},
    "断": {"viet": "ĐOẠN, ĐOÁN", "meaning_vi": "Như chữ [đoạn [斷]. Giản thể của chữ 斷", "meo": "", "reading": "た.つ ことわ.る さだ.める", "vocab": [("断", "だん", "sự không xảy ra"), ("断つ", "だんつ", "dứt .")]},
    "歯": {"viet": "XỈ", "meaning_vi": "Răng", "meo": "", "reading": "よわい は よわ.い よわい.する", "vocab": [("歯", "は", "răng"), ("乳歯", "にゅうし", "răng sữa")]},
    "神": {"viet": "THẦN", "meaning_vi": "Thiên thần.", "meo": "", "reading": "かみ かん- こう-", "vocab": [("神", "かみ", "chúa"), ("神主", "かんぬし", "người đứng đầu giáo phái Shinto .")]},
    "伸": {"viet": "THÂN", "meaning_vi": "Duỗi. Như dẫn thân [引伸] kéo duỗi ra.", "meo": "", "reading": "の.びる の.ばす の.べる の.す", "vocab": [("伸", "しん", "sắt"), ("伸す", "のばす", "sự căng ra")]},
    "保": {"viet": "BẢO", "meaning_vi": "Gánh vác, gánh lấy trách nhiệm gọi là bảo. Như bảo chứng [保證] nhận làm chứng, bảo hiểm [保險] nhận giúp đỡ lúc nguy hiểm, trung bảo [中保] người đứng giữa nhận trách nhiệm giới thiệu cả hai bên.", "meo": "", "reading": "たも.つ", "vocab": [("保", "ほ", "sự bảo đảm"), ("保つ", "たもつ", "giữ; bảo vệ; duy trì")]},
    # "意" → already in main DB
    "億": {"viet": "ỨC", "meaning_vi": "Ức. Mười vạn [萬] là một ức [億].", "meo": "", "reading": "オク", "vocab": [("億", "おく", "100 triệu"), ("一億", "いちおく", "một trăm triệu")]},
    "氷": {"viet": "BĂNG", "meaning_vi": "Tục dùng như chữ băng [冰].", "meo": "", "reading": "こおり ひ こお.る", "vocab": [("氷", "こおり", "đá (ăn)"), ("氷", "ひ", "băng")]},
    "永": {"viet": "VĨNH", "meaning_vi": "Lâu, dài, mãi mãi. Như vĩnh viễn [永遠] mãi mãi, vĩnh phúc [永福] điều may mắn được hưởng lâu dài.", "meo": "", "reading": "なが.い", "vocab": [("永々", "ひさし々", "mãi mãi"), ("永い", "ながい", "dài; dài lâu")]},
    "位": {"viet": "VỊ", "meaning_vi": "Ngôi, cái chỗ ngồi của mình được ở gọi là vị. Như địa vị [地位], tước vị [爵位], v.v.", "meo": "", "reading": "くらい ぐらい", "vocab": [("位", "くらい", "khoảng; chừng; cỡ độ; xấp xỉ; mức"), ("上位", "じょうい", "lớp trên; vị trí cao")]},
    "泣": {"viet": "KHẤP", "meaning_vi": "Khóc, khóc không ra tiếng gọi là khấp. Nguyễn Du [阮攸] : Bất tri tam bách dư niên hậu, Thiên hạ hà nhân khấp Tố Như [不知三百餘年後, 天下何人泣素如] (Độc Tiểu Thanh kí [讀小青記]) Không biết hơn ba trăm năm sau, Thiên hạ ai là người khóc Tố Như (*). $ (*) Tố Như [素如] là tên tự của Nguyễn Du [阮攸] (1765-1820).", "meo": "", "reading": "な.く", "vocab": [("泣き", "なき", "đang khóc"), ("泣く", "なく", "khóc")]},
    "活": {"viet": "HOẠT, QUẠT", "meaning_vi": "Sống, phàm những sự để nuôi sống đều gọi là sinh hoạt [生活].", "meo": "", "reading": "い.きる い.かす い.ける", "vocab": [("活劇", "かつげき", "kịch nói ."), ("活力", "かつりょく", "sức sống; sinh khí; sự tồn tại lâu dài")]},
    "昨": {"viet": "TẠC", "meaning_vi": "Hôm qua. Như tạc nhật [昨日] ngày hôm qua, tạc dạ [昨夜] đêm qua, tạc niên [昨年] năm ngoái, v.v.", "meo": "", "reading": "サク", "vocab": [("昨", "さく", "trước (năm"), ("昨今", "さっこん", "ngày nay; gần đây .")]},
    "級": {"viet": "CẤP", "meaning_vi": "Bậc, mỗi một bậc thềm gọi là một cấp. Lên thềm gọi là thập cấp [拾級].", "meo": "", "reading": "キュウ", "vocab": [("級", "きゅう", "bực"), ("一級", "いっきゅう", "bậc nhất .")]},
    "吸": {"viet": "HẤP", "meaning_vi": "Hút hơi vào. Đối lại với chữ hô [呼].", "meo": "", "reading": "す.う", "vocab": [("吸う", "すう", "bú"), ("吸入", "きゅうにゅう", "sự hô hấp; sự hít vào; sự hít thở; hô hấp; hít vào; hít thở")]},
    "全": {"viet": "TOÀN", "meaning_vi": "Xong, đủ.", "meo": "", "reading": "まった.く すべ.て", "vocab": [("全", "ぜん", "toàn bộ"), ("全う", "まっとう", "hoàn thành")]},
    "欲": {"viet": "DỤC", "meaning_vi": "Tham muốn.", "meo": "", "reading": "ほっ.する ほ.しい", "vocab": [("欲", "よく", "sự mong muốn; sự tham lam"), ("利欲", "りよく", "tính tham lam")]},
    "浴": {"viet": "DỤC", "meaning_vi": "Tắm. Như mộc dục [沐浴] tắm gội.", "meo": "", "reading": "あ.びる あ.びせる", "vocab": [("浴", "よく", "sự tắm"), ("入浴", "にゅうよく", "việc tắm táp .")]},
    "容": {"viet": "DUNG, DONG", "meaning_vi": "Bao dong chịu đựng. Như hưu hưu hữu dong [休休有容] lồng lộng có lượng bao dong, nghĩa là khí cục rộng lớn bao dong cả được mọi người. Cái vật gì chứa được bao nhiêu gọi là dong lượng [容量].", "meo": "", "reading": "い.れる", "vocab": [("容体", "ようだい", "tình trạng cơ thể; trạng thái cơ thể"), ("偉容", "いよう", "chân giá trị")]},
    "首": {"viet": "THỦ, THÚ", "meaning_vi": "Đầu. Như khể thủ [稽首] lạy dập đầu. Dân gọi là kiềm thủ [黔首] nói những kẻ trai trẻ tóc đen có thể gánh vác mọi việc cho nhà nước vậy.", "meo": "", "reading": "くび", "vocab": [("首", "くび", "cổ"), ("一首", "いちしゅ", "bài thơ")]},
    "負": {"viet": "PHỤ", "meaning_vi": "Cậy, cậy có chỗ tựa không sợ gọi là phụ. Như phụ ngung chi thế [負嵎之勢] cậy có cái thế đằng sau có chỗ tựa chắc, cậy tài khinh người gọi là tự phụ bất phàm [自負不凡].", "meo": "", "reading": "ま.ける ま.かす お.う", "vocab": [("負", "まけ", "không; phủ định"), ("負う", "おう", "mang; gánh vác; nợ; vác; khuân")]},
    "敗": {"viet": "BẠI", "meaning_vi": "Hỏng, đổ nát. Như vong quốc bại gia [亡國敗家] làm mất nước nát nhà. Đứa con làm hỏng nhà gọi là bại tử [敗子], nhục bại [肉敗] thịt đã thiu thối, bại diệp [敗葉] lá rụng, v.v.", "meo": "", "reading": "やぶ.れる", "vocab": [("敗る", "はいる", "sự thất bại"), ("不敗", "ふはい", "tính vô địch")]},
    "則": {"viet": "TẮC", "meaning_vi": "Phép. Nội các chế đồ khuôn mẫu gì đều gọi là tắc, nghĩa là để cho người coi đó mà bắt chước vậy. Như ngôn nhi vi thiên hạ tắc [言而為天下則] nói mà làm phép cho thiên hạ.", "meo": "", "reading": "のっと.る", "vocab": [("会則", "かいそく", "qui tắc của hội; quy tắc tổ chức; điều lệ hiệp hội"), ("党則", "とうそく", "quy tắc Đảng .")]},
    "側": {"viet": "TRẮC", "meaning_vi": "Bên. Như trắc diện [側面] mặt bên, trắc thất [側室] vợ lẽ.", "meo": "", "reading": "かわ がわ そば", "vocab": [("側", "かわ", "phía"), ("側", "がわ", "bề")]},
    "測": {"viet": "TRẮC", "meaning_vi": "Đo chiều sâu, nói rộng ra phàm sự đo lường đều gọi là trắc cả. Như bất trắc [不測] không lường được.", "meo": "", "reading": "はか.る", "vocab": [("測る", "はかる", "dò"), ("不測", "ふそく", "bất trắc .")]},
    # "員" → already in main DB
    "性": {"viet": "TÍNH", "meaning_vi": "Tính, là một cái lẽ chân chính trời bẩm phú cho người. Như tính thiện [性善] tính lành.", "meo": "", "reading": "さが", "vocab": [("性", "せい", "giới tính; giống"), ("両性", "りょうせい", "lưỡng tính .")]},
    "産": {"viet": "SẢN", "meaning_vi": "Một dạng của chữ sản [產].", "meo": "", "reading": "う.む う.まれる うぶ- む.す", "vocab": [("お産", "おさん", "việc sinh đẻ; sự ra đời; sinh nở; chuyển dạ"), ("産み", "うみ", "Sự sinh đẻ; sinh; sinh nở; sinh hạ; đẻ; thành lập")]},
    "等": {"viet": "ĐẲNG", "meaning_vi": "Bực. Như xuất giáng nhất đẳng [出降一等] (Luận ngữ [論語]) giáng xuống một bực, thượng đẳng [上等] bực trên nhất, trung đẳng [中等] bực giữa, hạ đẳng [下等] bực dưới nhất (hạng bét); v.v.", "meo": "", "reading": "ひと.しい など -ら", "vocab": [("等", "など", "Vân vân"), ("等々", "などなど", "Vân vân")]},
    "記": {"viet": "KÍ", "meaning_vi": "Nhớ, nhớ kĩ cho khỏi quên. Như kí tụng [記誦] học thuộc cho nhớ.", "meo": "", "reading": "しる.す", "vocab": [("記", "き", "sử biên niên; ký sự niên đại"), ("記す", "しるす", "đánh dấu .")]},
    # "起" → already in main DB
    "副": {"viet": "PHÓ", "meaning_vi": "Thứ hai. Như phó sứ [副使], phó lý [副里], v.v.", "meo": "", "reading": "フク", "vocab": [("副", "ふく", "phụ; phó"), ("副う", "ふくう", "bộ com lê")]},
    "福": {"viet": "PHÚC", "meaning_vi": "Phúc, những sự tốt lành đều gọi là phúc. Kinh Thi chia ra năm phúc : (1) Giàu [富] (2) Yên lành [安寧] (3) Thọ [壽] (4) Có đức tốt [攸好德] (5) Vui hết tuổi trời [考終命].", "meo": "", "reading": "フク", "vocab": [("福", "ふく", "hạnh phúc"), ("万福", "ばんぷく", "sức khỏe và hạnh phúc; vạn phúc .")]},
    "富": {"viet": "PHÚ", "meaning_vi": "Giàu.", "meo": "", "reading": "と.む とみ", "vocab": [("富", "とみ", "của cải; tài sản"), ("富む", "とむ", "giàu có")]},
    "支": {"viet": "CHI", "meaning_vi": "Chi, thứ. Như trưởng chi [長支] chi trưởng, chi tử [支子] con thứ, v.v.", "meo": "", "reading": "ささ.える つか.える か.う", "vocab": [("支え", "ささえ", "sự ủng hộ"), ("支出", "ししゅつ", "sự chi ra; sự xuất ra; mức chi ra .")]},
    "技": {"viet": "KĨ", "meaning_vi": "Nghề. Như tràng kĩ [長技] nghề tài, mạt kĩ [末技] nghề mạt hạng, v.v.", "meo": "", "reading": "わざ", "vocab": [("技", "わざ", "kỹ năng; kỹ thuật"), ("余技", "よぎ", "công việc phụ; việc lặt vặt")]},
    "相": {"viet": "TƯƠNG, TƯỚNG", "meaning_vi": "Cùng. Như bỉ thử tương ái [彼此相愛] đây đấy cùng yêu nhau.", "meo": "", "reading": "あい-", "vocab": [("相", "あい", "cùng nhau; ổn định; hòa hợp"), ("相", "そう", "dáng; trạng thái")]},
    "箱": {"viet": "TƯƠNG, SƯƠNG", "meaning_vi": "Cái hòm xe, trong xe đóng một cái ngăn để chứa đồ gọi là xa tương [車箱].", "meo": "", "reading": "はこ", "vocab": [("箱", "はこ", "hòm"), ("大箱", "だいばこ", "hộp lớn .")]},
    "想": {"viet": "TƯỞNG", "meaning_vi": "Tưởng tượng. Lòng muốn cái gì nghĩ vào cái ấy gọi là tưởng.", "meo": "", "reading": "おも.う", "vocab": [("想", "そう", "quan niệm; ý niệm; ý tưởng; suy nghĩ ."), ("想い", "おもい", "sự suy nghĩ")]},
    "拾": {"viet": "THẬP, THIỆP, KIỆP", "meaning_vi": "Nhặt nhạnh. Nguyễn Du [阮攸] : Hành ca thập tuệ thì [行歌拾穗時] (Vinh Khải Kì [榮棨期]) Vừa ca vừa mót lúa.", "meo": "", "reading": "ひろ.う", "vocab": [("拾", "じゅう", "thập"), ("拾う", "ひろう", "lượm")]},
    "給": {"viet": "CẤP", "meaning_vi": "Đủ dùng. Như gia cấp nhân túc [家給人足] nhà no người đủ.", "meo": "", "reading": "たま.う たも.う -たま.え", "vocab": [("給", "きゅう", "lương; tiền công"), ("給う", "たまう", "nhận")]},
    "血": {"viet": "HUYẾT", "meaning_vi": "Máu.", "meo": "", "reading": "ち", "vocab": [("血", "ち", "huyết"), ("充血", "じゅうけつ", "sung huyết .")]},
    "塩": {"viet": "DIÊM", "meaning_vi": "Muối", "meo": "", "reading": "しお", "vocab": [("塩", "えん", "muối"), ("塩", "しお", "muối")]},
    "温": {"viet": "ÔN, UẨN", "meaning_vi": "Giản thể của chữ 溫", "meo": "", "reading": "あたた.か あたた.かい あたた.まる あたた.める ぬく", "vocab": [("温々", "あつし々", "tiện lợi"), ("温い", "ぬるい", "nguội; âm ấm")]},
    "仲": {"viet": "TRỌNG", "meaning_vi": "Giữa. Như tháng hai gọi là trọng xuân [仲春] giữa mùa xuân, em thứ hai là trọng đệ [仲第] v.v.", "meo": "", "reading": "なか", "vocab": [("仲", "なか", "quan hệ"), ("不仲", "ふなか", "sự bất hoà; mối bất hoà")]},
    "判": {"viet": "PHÁN", "meaning_vi": "Lìa rẽ. Như phán duệ [判袂] chia tay mỗi người một ngả. Ôn Đình Quân [溫庭筠] : Dạ văn mãnh vũ phán hoa tận [夜聞猛雨判花盡] Đêm nghe mưa mạnh làm tan tác hết các hoa.", "meo": "", "reading": "わか.る", "vocab": [("判", "はん", "con dấu; triện"), ("判", "ばん", "kích cỡ")]},
    "険": {"viet": "HIỂM", "meaning_vi": "Nguy hiểm, mạo hiểm, hiểm ác", "meo": "", "reading": "けわ.しい", "vocab": [("保険", "ほけん", "sự bảo hiểm"), ("冒険", "ぼうけん", "sự mạo hiểm")]},
    "検": {"viet": "KIỂM", "meaning_vi": "Kiểm tra", "meo": "", "reading": "しら.べる", "vocab": [("検事", "けんじ", "công tố viên; ủy viên công tố; kiểm sát viên"), ("検体", "けんたい", "mẫu")]},
    "皮": {"viet": "BÌ", "meaning_vi": "Da. Da giống thú còn có lông gọi là bì [皮], không có lông gọi là cách [革]. Nguyễn Du [阮攸] : Mao ám bì can sấu bất câm [毛暗皮乾瘦不禁] (Thành hạ khí mã [城下棄馬]) Lông nám da khô gầy không thể tả.", "meo": "", "reading": "かわ", "vocab": [("皮", "かわ", "da bì"), ("上皮", "じょうひ", "biểu bì")]},
    "彼": {"viet": "BỈ", "meaning_vi": "Bên kia. Là tiếng trái lại với chữ thử. Như bất phân bỉ thử [不分彼此] chẳng phân biệt được đấy với đây.", "meo": "", "reading": "かれ かの か.の", "vocab": [("彼", "かれ", "anh ta"), ("彼の", "あの", "cái đó; chỗ đó")]},
    # "仕" → already in main DB
    "必": {"viet": "TẤT", "meaning_vi": "Ắt hẳn, lời nói quyết định. Như tất nhiên [必然] sự tất thế.", "meo": "", "reading": "かなら.ず", "vocab": [("必ず", "かならず", "nhất định; tất cả"), ("必中", "ひっちゅう", "sự đánh đích .")]},
    "湯": {"viet": "THANG, SƯƠNG, THÃNG", "meaning_vi": "Nước nóng.", "meo": "", "reading": "ゆ", "vocab": [("湯", "ゆ", "nước sôi"), ("お湯", "おゆ", "nước nóng")]},
    "陽": {"viet": "DƯƠNG", "meaning_vi": "Phần dương, khí dương. Trái lại với chữ âm [陰]. Xem lại chữ âm [陰].", "meo": "", "reading": "ひ", "vocab": [("陽", "よう", "mặt trời; ánh sáng mặt trời ."), ("陽光", "ようこう", "ánh sáng mặt trời; ánh nắng .")]},
    "幼": {"viet": "ẤU", "meaning_vi": "Nhỏ bé, non nớt. Trẻ bé gọi là ấu trĩ [幼稚]. Học thức còn ít cũng gọi là ấu trĩ, nghĩa là trình độ còn non như trẻ con vậy.", "meo": "", "reading": "おさな.い", "vocab": [("幼い", "おさない", "trẻ con; ngây thơ"), ("幼児", "ようじ", "hài đồng")]},
    "粉": {"viet": "PHẤN", "meaning_vi": "Bột gạo, phấn gạo.", "meo": "", "reading": "デシメートル こ こな", "vocab": [("粉", "こな", "bột mì; bột"), ("粉体", "こなたい", "bột; bụi")]},
    "貧": {"viet": "BẦN", "meaning_vi": "Nghèo. Như bần sĩ [貧士] học trò nghèo.", "meo": "", "reading": "まず.しい", "vocab": [("貧", "ひん", "sự nghèo nàn; cảnh nghèo nàn"), ("貧乏", "びんぼう", "bần cùng")]},
    "菜": {"viet": "THÁI", "meaning_vi": "Rau. Thứ rau cỏ nào ăn được đều gọi là thái. Người đói phải ăn rau trừ bữa nên gọi là thái sắc [菜色].", "meo": "", "reading": "な", "vocab": [("菜", "な", "rau cỏ ."), ("お菜", "おかず", "món ăn thêm; thức ăn kèm; món nhắm; đồ nhắm; nhắm; món nhậu; đồ nhậu")]},
    "良": {"viet": "LƯƠNG", "meaning_vi": "Lành, tính chất thuần tốt bền giữ không đổi gọi là lương. Như trung lương [忠良], hiền lương [賢良], v.v. Cái tâm thuật của người gọi là thiên lương [天良], tục gọi là lương tâm [良心]. Tục gọi con nhà thanh bạch, không có tiếng tăm gì xấu là lương gia tử đệ [良家子弟] con em nhà lương thiện. Cô đầu nhà thổ giũ sổ về làm ăn lương thiện gọi là tòng lương [從良].", "meo": "", "reading": "よ.い -よ.い い.い -い.い", "vocab": [("良", "りょう", "tốt"), ("良い", "いい", "tốt; đẹp; đúng")]},
    "娘": {"viet": "NƯƠNG", "meaning_vi": "Nàng. Con gái trẻ tuổi gọi là nương tử [娘子] hay cô nương [姑娘] cô nàng.", "meo": "", "reading": "むすめ こ", "vocab": [("娘", "むすめ", "con gái ."), ("娘婿", "むすめむこ", "con gái nuôi .")]},
    "根": {"viet": "CĂN", "meaning_vi": "Rễ cây.", "meo": "", "reading": "ね -ね", "vocab": [("根", "ね", "cội"), ("主根", "しゅね", "rễ cái")]},
    "退": {"viet": "THỐI, THOÁI", "meaning_vi": "Lui. Như thối binh [退兵] lui binh.", "meo": "", "reading": "しりぞ.く しりぞ.ける ひ.く の.く の.ける ど.く", "vocab": [("退く", "どく", "rút"), ("退く", "ひく", "rút")]},
    "達": {"viet": "ĐẠT", "meaning_vi": "Suốt. Như tứ thông bát đạt [四通八達] thông cả bốn mặt suốt cả tám phía.", "meo": "", "reading": "-たち", "vocab": [("達", "たち", "những"), ("達し", "たっし", "sự báo")]},
    "芸": {"viet": "NGHỆ", "meaning_vi": "Một thứ cỏ thơm (mần tưới); thường gọi là cỏ vân hương [芸香], lấy lá hoa nó gấp vào sách thì khỏi mọt. Vì thế nên gọi quyển sách là vân biên [芸編]. Nguyễn Du [阮攸] : Vân song tằng kỷ nhiễm thư hương [芸窗曾幾染書香] (Điệp tử thư trung [蝶死書中]) Nơi cửa sổ trồng cỏ vân đã từng bao lần đượm mùi hương sách vở.", "meo": "", "reading": "う.える のり わざ", "vocab": [("芸", "げい", "tài khéo léo"), ("一芸", "いちげい", "tài khéo léo")]},
    "絵": {"viet": "HỘI", "meaning_vi": "Hội họa", "meo": "", "reading": "カイ エ", "vocab": [("絵", "え", "bức tranh; tranh"), ("下絵", "したえ", "tranh đả kích")]},
    # "転" → already in main DB
    "伝": {"viet": "TRUYỀN", "meaning_vi": "Truyền đạt, truyền động", "meo": "", "reading": "つた.わる つた.える つた.う つだ.う -づた.い つて", "vocab": [("伝", "つて", "ở giữa"), ("伝う", "つたう", "đi cùng; đi theo")]},
    "効": {"viet": "HIỆU", "meaning_vi": "Công hiệu. Tục dùng như chữ [效].", "meo": "", "reading": "き.く ききめ なら.う", "vocab": [("効", "こう", "tính có hiệu quả; hiệu lực"), ("効く", "きく", "có tác dụng; có hiệu quả; có ảnh hưởng; có kết quả")]},
    "算": {"viet": "TOÁN", "meaning_vi": "Số vật. Như vô toán [無算] rất nhiều không tính xiết.", "meo": "", "reading": "そろ", "vocab": [("乗算", "じょうさん", "phép nhân ."), ("予算", "よさん", "dự toán")]},
    "泊": {"viet": "BẠC, PHÁCH", "meaning_vi": "Ghé vào, đỗ thuyền bên bờ.", "meo": "", "reading": "と.まる と.める", "vocab": [("泊る", "とまる", "ở"), ("一泊", "いっぱく", "một đêm")]},
    "優": {"viet": "ƯU", "meaning_vi": "Nhiều. Như ưu ác [優渥] thừa thãi.", "meo": "", "reading": "やさ.しい すぐ.れる まさ.る", "vocab": [("優", "ゆう", "hiền lành"), ("優に", "ゆうに", "thoải mái")]},
    "線": {"viet": "TUYẾN", "meaning_vi": "Chỉ khâu.", "meo": "", "reading": "すじ", "vocab": [("線", "せん", "đường dây (điện thoại); đường ray; dây dẫn; đường"), ("x線", "えっくすせん", "tia chụp Xquang; tia X quang")]},
    "願": {"viet": "NGUYỆN", "meaning_vi": "Muốn. Lòng mong cầu gọi là tâm nguyện [心願]. Đem cái quyền lợi mình muốn được hưởng mà yêu cầu pháp luật định cho gọi là thỉnh nguyện [請願].", "meo": "", "reading": "ねが.う -ねがい", "vocab": [("願", "ねがい", "kinh cầu nguyện"), ("願い", "ねがい", "yêu cầu; nguyện cầu; mong ước .")]},
    "仏": {"viet": "PHẬT", "meaning_vi": "Phật giáo", "meo": "", "reading": "ほとけ", "vocab": [("仏", "ぶつ", "Phật thích ca; đạo phật"), ("仏", "ほとけ", "con người nhân từ")]},
    "身": {"viet": "THÂN", "meaning_vi": "cơ thể, bản thân, thân phận", "meo": "Chữ Thân (身) trông như một người đang đứng thẳng, với cái đầu ở trên và phần thân mình ở dưới. Hãy liên tưởng đến 'thân thể' của con người.", "reading": "み", "vocab": [("身体", "しんたい", "cơ thể"), ("自分", "じぶん", "bản thân")]},
    "続": {"viet": "TỤC", "meaning_vi": "Tiếp tục", "meo": "", "reading": "つづ.く つづ.ける つぐ.ない", "vocab": [("続々", "ぞくぞく", "sự liên tục; sự kế tiếp; việc cái này tiếp theo cái khác"), ("続き", "つづき", "sự tiếp tục; sự tiếp diễn")]},
    "情": {"viet": "TÌNH", "meaning_vi": "Tình, cái tình đã phát hiện ra ngoài. Như mừng, giận, thương, sợ, yêu, ghét, muốn gọi là thất tình.", "meo": "", "reading": "なさ.け", "vocab": [("情", "じょう", "cảm xúc; tình cảm; cảm giác ."), ("情け", "なさけ", "lòng trắc ẩn; sự cảm thông")]},
    "争": {"viet": "TRANH, TRÁNH", "meaning_vi": "Giản thể của chữ [爭]", "meo": "", "reading": "あらそ.う いか.でか", "vocab": [("争い", "あらそい", "sự tranh giành; sự đua tranh; sự đánh nhau; mâu thuẫn; chiến tranh; cuộc chiến; xung đột; tranh chấp"), ("争う", "あらそう", "gây gổ")]},
    "庫": {"viet": "KHỐ", "meaning_vi": "Cái kho. Chỗ để chứa đồ binh khí của nhà nước. Chỗ để đồ cũng gọi là khố.", "meo": "", "reading": "くら", "vocab": [("倉庫", "そうこ", "kho hàng"), ("入庫", "にゅうこ", "nhập kho")]},
    "軍": {"viet": "QUÂN", "meaning_vi": "Quân lính. Như lục quân [陸軍] quân bộ, hải quân [海軍] quân thủy, ngày xưa vua có sáu cánh quân, mỗi cánh quân có 125 000 quân. Phép binh bây giờ thì hai sư đoàn gọi là một cánh quân.", "meo": "", "reading": "いくさ", "vocab": [("軍", "ぐん", "quân đội; đội quân"), ("一軍", "いちぐん", "quân đội")]},
    "連": {"viet": "LIÊN", "meaning_vi": "Liền. Hai bên liền tiếp nhau gọi là liên.", "meo": "", "reading": "つら.なる つら.ねる つ.れる -づ.れ", "vocab": [("連", "れん", "nhóm; xê ri"), ("連む", "れんむ", "nước chiếu tướng")]},
    "求": {"viet": "CẦU", "meaning_vi": "Tìm, phàm muốn được cái gì mà hết lòng tìm tòi cho được đều gọi là cầu. Như sưu cầu [搜求] lục tìm, nghiên cầu [研求] nghiền tìm, v.v.", "meo": "", "reading": "もと.める", "vocab": [("求む", "もとむ", "sự thiếu"), ("求め", "もとめ", "lời thỉnh cầu")]},
    "球": {"viet": "CẦU", "meaning_vi": "Cái khánh ngọc.", "meo": "", "reading": "たま", "vocab": [("球", "きゅう", "quả cầu; cầu; hình tròn"), ("球", "たま", "quả bóng; hình cầu; ngọc; ngọc trai; hạt ngọc; bóng đèn; viên đạn .")]},
    "救": {"viet": "CỨU", "meaning_vi": "Ngăn, cản lại. Luận Ngữ [論語] : Quý thị lữ ư Thái Sơn. Tử vị Nhiễm Hữu viết : Nhữ phất năng cứu dữ ? Đối viết : Bất năng [季氏旅於泰山. 子謂冉有曰 : 女弗能救與 ? 對曰 : 不能] (Bát dật [八佾]) Họ Quý tế lữ ở núi Thái Sơn. Khổng Tử hỏi Nhiễm Hữu rằng : Anh không ngăn được sao ? Nhiễm Hữu đáp : Không được. Chú thích (1). $ (1) Ý nói, theo lễ thì vua Lỗ mới có quyền tế lữ, Họ Quý chỉ là một quan đại phu đã tiếm lễ.", "meo": "", "reading": "すく.う", "vocab": [("救い", "すくい", "sự giúp đỡ; sự cứu giúp; sự cứu tế ."), ("救う", "すくう", "cứu giúp; cứu tế; cứu trợ")]},
    "種": {"viet": "CHỦNG, CHÚNG", "meaning_vi": "Giống thóc.", "meo": "", "reading": "たね -ぐさ", "vocab": [("種", "しゅ", "chủng"), ("種", "たね", "hạt; hạt giống; thể loại; nhiều thứ")]},
    # "注" → already in main DB
    "柱": {"viet": "TRỤ, TRÚ", "meaning_vi": "Cái cột.", "meo": "", "reading": "はしら", "vocab": [("柱", "はしら", "cột"), ("中柱", "なかばしら", "Cột giữa; trụ giữa .")]},
    "辞": {"viet": "TỪ", "meaning_vi": "Tục dùng như chữ từ [辭].", "meo": "", "reading": "や.める いな.む", "vocab": [("辞す", "じす", "ký tên lại[ri'zain]"), ("世辞", "せじ", "sự tâng bốc; sự tán dương; sự ca tụng .")]},
    "幸": {"viet": "HẠNH", "meaning_vi": "May, hạnh phúc. Sự gì đáng bị thiệt mà lại thoát gọi là hạnh.", "meo": "", "reading": "さいわ.い さち しあわ.せ", "vocab": [("幸", "さち", "sự may mắn; hạnh phúc ."), ("幸い", "さいわい", "hân hạnh")]},
    "反": {"viet": "PHẢN, PHIÊN", "meaning_vi": "Trái, đối lại với chữ chính [正]. Bên kia mặt phải gọi là mặt trái.", "meo": "", "reading": "そ.る そ.らす かえ.す かえ.る -かえ.る", "vocab": [("反", "たん", "tan"), ("反", "はん", "mặt trái; mặt đối diện .")]},
    "返": {"viet": "PHẢN", "meaning_vi": "Trả lại.", "meo": "", "reading": "かえ.す -かえ.す かえ.る -かえ.る", "vocab": [("返す", "かえす", "trả"), ("返る", "かえる", "trở lại; trở về")]},
    "坂": {"viet": "PHẢN", "meaning_vi": "Sườn núi.", "meo": "", "reading": "さか", "vocab": [("坂", "さか", "cái dốc"), ("下坂", "しもさか", "dốc xuống")]},
    "板": {"viet": "BẢN", "meaning_vi": "Ván, mảnh mỏng. Như mộc bản [木板] tấm ván, đồng bản [銅板] lá đồng, ngày xưa gọi cái hốt là thủ bản [手板], tờ chiếu là chiếu bản [詔板] cũng do nghĩa ấy.", "meo": "", "reading": "いた", "vocab": [("板", "いた", "tấm ván"), ("板", "ばん", "bản .")]},
    "報": {"viet": "BÁO", "meaning_vi": "Báo trả, thù đáp lại. Nguyễn Trãi [阮廌] : Quốc ân vị báo lão kham liên [國恩未報老堪憐] (Hải khẩu dạ bạc hữu cảm [海口夜泊有感]) Ơn nước chưa đáp đền mà đã già, thật đáng thương.", "meo": "", "reading": "むく.いる", "vocab": [("報", "ほう", "báo cáo"), ("報い", "むくい", "sự thưởng")]},
    "常": {"viet": "THƯỜNG", "meaning_vi": "Thường (lâu mãi). Như vô thường [無常] nghĩa là không chắc chắn, thay đổi.", "meo": "", "reading": "つね とこ-", "vocab": [("常", "とこ", "sự vô cùng; sự vô tận"), ("常", "とわ", "Tính vĩnh hằng; tính vĩnh viễn; tính bất tử .")]},
    "賞": {"viet": "THƯỞNG", "meaning_vi": "Thưởng, thưởng cho kẻ có công.", "meo": "", "reading": "ほ.める", "vocab": [("賞", "しょう", "giải thưởng; giải"), ("賞与", "しょうよ", "thưởng; giải thưởng; tiền thưởng .")]},
    # "用" → already in main DB
    "痛": {"viet": "THỐNG", "meaning_vi": "Đau đớn, đau xót. Như thống khổ [痛苦] đau khổ.", "meo": "", "reading": "いた.い いた.む いた.ましい いた.める", "vocab": [("痛い", "いたい", "đau; đau đớn"), ("痛み", "いたみ", "cơn đau")]},
    "備": {"viet": "BỊ", "meaning_vi": "Đủ.", "meo": "", "reading": "そな.える そな.わる つぶさ.に", "vocab": [("備え", "そなえ", "sự soạn"), ("備に", "つぶさに", "hoàn toàn")]},
    "告": {"viet": "CÁO, CỐC", "meaning_vi": "Bảo, bảo người trên gọi là cáo.", "meo": "", "reading": "つ.げる", "vocab": [("お告", "おつげ", "lời tiên đoán; lời tiên tri; lời sấm truyền"), ("予告", "よこく", "sự báo trước; linh cảm; điềm báo trước")]},
    "造": {"viet": "TẠO, THÁO", "meaning_vi": "Gây nên, làm nên. Như tạo phúc nhất phương [造福一方] làm nên phúc cho cả một phương. Tạo nghiệt vô cùng [造孽無窮] gây nên mầm vạ vô cùng, v.v.", "meo": "", "reading": "つく.る つく.り -づく.り", "vocab": [("造り", "つくり", "kết cấu"), ("造る", "つくる", "cắt tỉa (cây)")]},
    "材": {"viet": "TÀI", "meaning_vi": "Gỗ dùng được, phàm vật gì của trời sinh mà có thể lấy để dùng được đều gọi là tài. Như kim, mộc, thủy, hỏa, thổ [金木水火土] gọi là ngũ tài [五材].", "meo": "", "reading": "ザイ", "vocab": [("丸材", "まるざい", "khúc gỗ mới đốn"), ("人材", "じんざい", "nhân tài .")]},
    "筆": {"viet": "BÚT", "meaning_vi": "Cái bút.", "meo": "", "reading": "ふで", "vocab": [("筆", "ふで", "bút"), ("主筆", "しゅひつ", "chủ bút .")]},
    "律": {"viet": "LUẬT", "meaning_vi": "Luật lữ, cái đồ ngày xưa dùng để xét tiếng tăm.", "meo": "", "reading": "リツ リチ レツ", "vocab": [("律", "りつ", "lời răn dạy; nguyên tắc"), ("一律", "いちりつ", "sự ngang bằng")]},
    # "建" → already in main DB
    "結": {"viet": "KẾT", "meaning_vi": "Thắt nút dây. Đời xưa chưa có chữ, cứ mỗi việc thắt một nút dây để làm ghi gọi là kết thằng chi thế [結繩之世] hay kết thằng kí sự [結繩記事]. Tết dây thao đỏ lại để làm đồ trang sức cũng gọi là kết.", "meo": "", "reading": "むす.ぶ ゆ.う ゆ.わえる", "vocab": [("結う", "ゆう", "nối; buộc; ken ."), ("結び", "むすび", "sự liên kết; sự kết thúc")]},
    "捨": {"viet": "XÁ, XẢ", "meaning_vi": "Vất bỏ. Như xả thân hoằng đạo [捨身弘道] bỏ mình làm việc đạo.", "meo": "", "reading": "す.てる", "vocab": [("取捨", "しゅしゃ", "sự chọn lọc"), ("喜捨", "きしゃ", "sự bố thí; bố thí")]},
    "祝": {"viet": "CHÚC, CHÚ", "meaning_vi": "Khấn. Như tâm trung mặc mặc đảo chúc [心中默默禱祝] trong bụng ngầm khấn nguyện.", "meo": "", "reading": "いわ.う", "vocab": [("祝い", "いわい", "chúc tụng"), ("祝う", "いわう", "ăn mừng; chúc; chúc mừng .")]},
    "税": {"viet": "THUẾ, THỐI, THOÁT", "meaning_vi": "Giản thể của chữ 稅", "meo": "", "reading": "ゼイ", "vocab": [("税", "ぜい", "thuế ."), ("免税", "めんぜい", "sự miễn thuế .")]},
    "説": {"viet": "THUYẾT, DUYỆT, THUẾ", "meaning_vi": "Dị dạng của chữ [说].", "meo": "", "reading": "と.く", "vocab": [("説", "せつ", "thuyết"), ("説く", "とく", "giải thích; biện hộ; bào chữa")]},
    "量": {"viet": "LƯỢNG, LƯƠNG", "meaning_vi": "Đồ đong. Các cái như cái đấu, cái hộc dùng để đong đều gọi là lượng cả.", "meo": "", "reading": "はか.る", "vocab": [("量", "りょう", "khối lượng"), ("量り", "はかり", "sự đo")]},
    "童": {"viet": "ĐỒNG", "meaning_vi": "Trẻ thơ. Mười lăm tuổi trở lại gọi là đồng tử [童子], mười lăm tuổi trở lên gọi là thành đồng [成童]. Luận ngữ [論語] : Quán giả ngũ lục nhân, đồng tử lục thất nhân, dục hồ Nghi, phong hồ Vũ Vu, vịnh nhi quy [冠者五六人，童子六七人，浴乎沂，風乎舞雩，詠而歸] Năm sáu người vừa tuổi đôi mươi, với sáu bảy đồng tử, dắt nhau đi tắm ở sông Nghi rồi lên hứng mát ở nền Vũ Vu, vừa đi vừa hát kéo nhau về nhà.", "meo": "", "reading": "わらべ", "vocab": [("童", "わらべ", "đứa trẻ; nhi đồng; trẻ nhỏ ."), ("児童", "じどう", "nhi đồng")]},
    # "者" → already in main DB
    # "代" → already in main DB
    "袋": {"viet": "ĐẠI", "meaning_vi": "Cái đẫy. Như tửu nang phạn đại [酒囊飯袋] giá áo túi cơm.", "meo": "", "reading": "ふくろ", "vocab": [("袋", "ふくろ", "bì; bao; túi; phong bao"), ("お袋", "おふくろ", "mẹ; mẹ đẻ")]},
    # "貸" → already in main DB
    "苦": {"viet": "KHỔ", "meaning_vi": "Đắng. Như khổ qua [苦瓜] mướp đắng.", "meo": "", "reading": "くる.しい -ぐる.しい くる.しむ くる.しめる にが.い にが.る", "vocab": [("苦い", "にがい", "đắng"), ("苦さ", "にがさ", "vị đắng")]},
    "個": {"viet": "CÁ", "meaning_vi": "Tục dùng như chữ  cá  [箇]. Dị dạng của chữ 个", "meo": "", "reading": "コ カ", "vocab": [("個", "こ", "cái; chiếc"), ("個々", "ここ", "từng ... một; từng")]},
    "受": {"viet": "THỤ", "meaning_vi": "Chịu nhận lấy. Người này cho, người kia chịu lấy gọi là thụ thụ [受受].", "meo": "", "reading": "う.ける -う.け う.かる", "vocab": [("受け", "うけ", "người giữ"), ("享受", "きょうじゅ", "sự hưởng thụ; hưởng thụ; nhận; hưởng")]},
    "授": {"viet": "THỤ", "meaning_vi": "Cho, trao cho.", "meo": "", "reading": "さず.ける さず.かる", "vocab": [("授与", "じゅよ", "việc trao tặng; trao tặng ."), ("授乳", "じゅにゅう", "sự chăm sóc bệnh nhân")]},
    "点": {"viet": "ĐIỂM", "meaning_vi": "Tục dùng như chữ điểm [點].", "meo": "", "reading": "つ.ける つ.く た.てる さ.す とぼ.す とも.す ぼち", "vocab": [("点", "てん", "điểm"), ("点々", "てんてん", "rời rạc")]},
    "調": {"viet": "ĐIỀU, ĐIỆU", "meaning_vi": "Điều hòa. Như điều quân [調勻] hòa đều nhau.", "meo": "", "reading": "しら.べる しら.べ ととの.う ととの.える", "vocab": [("調", "ちょう", "hắc ín"), ("調う", "ととのう", "sẵn sàng")]},
    "可": {"viet": "KHẢ, KHẮC", "meaning_vi": "Ưng cho.", "meo": "", "reading": "-べ.き -べ.し", "vocab": [("可", "か", "có thể; khả; chấp nhận; được phép"), ("不可", "ふか", "không kịp; không đỗ")]},
    "河": {"viet": "HÀ", "meaning_vi": "Sông. Hà Hán [河漢] là sông Thiên Hà ở trên trời, cao xa vô cùng, cho nên những kẻ nói khoác không đủ tin gọi là hà hán.", "meo": "", "reading": "かわ", "vocab": [("河", "かわ", "sông; dòng sông"), ("河原", "かわはら", "bãi bồi ven sông")]},
    "局": {"viet": "CỤC", "meaning_vi": "Cuộc, bộ phận. Chia làm bộ phận riêng đều gọi là cục. Như việc quan chia riêng từng bọn để làm riêng từng việc gọi là chuyên cục [専局], cho nên người đương sự gọi là đương cục [當局] người đang cuộc, cục nội [局內] trong cuộc, cục ngoại [局外] ngoài cuộc, v.v.", "meo": "", "reading": "つぼね", "vocab": [("局", "きょく", "cục (quản lý); đơn vị; ty"), ("事局", "こときょく", "vị trí")]},
    "翌": {"viet": "DỰC", "meaning_vi": "Ngày mai, kỳ tới. Dực nhật [翌日] ngày mai, dực niên [翌年] năm tới.", "meo": "", "reading": "ヨク", "vocab": [("翌々", "よくよく", "thận trọng"), ("翌年", "よくねん", "năm sau; năm tiếp theo")]},
    "紹": {"viet": "THIỆU", "meaning_vi": "Nối, con em nối được nghiệp của ông cha, gọi là khắc thiệu cơ cừu [克紹箕裘].", "meo": "", "reading": "ショウ", "vocab": [("紹介", "しょうかい", "sự giới thiệu; giới thiệu ."), ("紹介する", "しょうかい", "giới thiệu")]},
    "兵": {"viet": "BINH", "meaning_vi": "Đồ binh. Các đồ như súng ống, giáo mác đều gọi là binh khí [兵器]. Lính, phép binh bây giờ chia làm ba : (1) hạng thường bị ; (2) tục bị ; (3) hậu bị. Hiện đang ở lính gọi là thường bị binh, hết hạn ba năm về nhà ; có việc lại ra là tục bị binh ; lại đang hạn ba năm nữa rồi về là hậu bị binh, lại hết bốn năm cho về hưu hẳn, lại như dân thường.", "meo": "", "reading": "つわもの", "vocab": [("兵", "つわもの", "lính"), ("兵乱", "へいらん", "chiến tranh")]},
    "移": {"viet": "DI, DỊ, SỈ", "meaning_vi": "Dời đi. Nguyễn Du [阮攸] : Tào thị vu thử di Hán đồ [曹氏于此移漢圖] (Cựu Hứa đô [舊許都]) Họ Tào dời đô nhà Hán đến đây.", "meo": "", "reading": "うつ.る うつ.す", "vocab": [("移す", "うつす", "dọn đi"), ("移り", "うつり", "sự đổi")]},
    "夢": {"viet": "MỘNG, MÔNG", "meaning_vi": "Chiêm bao, nằm mê. Nguyễn Trãi [阮廌] : Mộng ki hoàng hạc thướng tiên đàn [夢騎黃鶴上仙壇] (Mộng sơn trung [夢山中]) Mơ thấy cưỡi hạc vàng bay lên đàn tiên.", "meo": "", "reading": "ゆめ ゆめ.みる くら.い", "vocab": [("夢", "ゆめ", "chiêm bao"), ("一夢", "いちゆめ", "giấc mơ")]},
    "秒": {"viet": "MIỂU", "meaning_vi": "Tua lúa.", "meo": "", "reading": "ビョウ", "vocab": [("秒", "びょう", "giây"), ("分秒", "ふんびょう", "chốc")]},
    "央": {"viet": "ƯƠNG", "meaning_vi": "Ở giữa. Như trung ương [中央] chỗ chính giữa, chỉ nơi tập trung quan trọng nhất..", "meo": "", "reading": "オウ", "vocab": [("中央", "ちゅうおう", "trung ương"), ("震央", "しんおう", "tâm động đất .")]},
    "決": {"viet": "QUYẾT", "meaning_vi": "Khơi, tháo.", "meo": "", "reading": "き.める -ぎ.め き.まる さ.く", "vocab": [("決", "けつ", "sự giải quyết ; sự phân xử"), ("決め", "きめ", "hiệp định")]},
    "窓": {"viet": "SONG", "meaning_vi": "Tục dùng như chữ song [窗].", "meo": "", "reading": "まど てんまど けむだし", "vocab": [("窓", "まど", "cửa sổ"), ("出窓", "でまど", "Cửa sổ xây lồi ra ngoài .")]},
    "供": {"viet": "CUNG", "meaning_vi": "Bầy, đặt. Như cung trướng [供帳] bỏ màn sẵn cho người ngủ.", "meo": "", "reading": "そな.える とも -ども", "vocab": [("供", "とも", "sự cùng nhau ."), ("供え", "そなえ", "sự biếu")]},
    "選": {"viet": "TUYỂN, TUYẾN", "meaning_vi": "Chọn. Tới trong số nhiều mà kén chọn lấy một số tốt đẹp gọi là tuyển. Như tinh tuyển [精選] chọn kỹ.", "meo": "", "reading": "えら.ぶ", "vocab": [("選", "せん", "sự lựa chọn"), ("選ぶ", "えらぶ", "bầu")]},
    "帯": {"viet": "ĐỚI", "meaning_vi": "Nhiệt đới, ôn đới", "meo": "", "reading": "お.びる おび", "vocab": [("帯", "おび", "đai"), ("帯", "たい", "việc mang (tính dẫn")]},
    "婦": {"viet": "PHỤ", "meaning_vi": "Vợ.", "meo": "", "reading": "よめ", "vocab": [("主婦", "しゅふ", "vợ ."), ("婦人", "ふじん", "phụ nữ .")]},
    "初": {"viet": "SƠ", "meaning_vi": "Mới, trước.", "meo": "", "reading": "はじ.め はじ.めて はつ はつ- うい- -そ.める -ぞ.め", "vocab": [("初", "はつ", "cái đầu tiên; cái mới"), ("初め", "はじめ", "ban đầu; lần đầu; khởi đầu")]},
    "覚": {"viet": "GIÁC", "meaning_vi": "Cảm giác,", "meo": "", "reading": "おぼ.える さ.ます さ.める さと.る", "vocab": [("覚え", "おぼえ", "ghi nhớ; nhớ"), ("覚り", "さとり", "sự hiểu biết")]},
    "現": {"viet": "HIỆN", "meaning_vi": "Hiển hiện, rõ ràng.", "meo": "", "reading": "あらわ.れる あらわ.す うつつ うつ.つ", "vocab": [("現す", "あらわす", "biểu lộ"), ("現に", "げんに", "thực sự là; thực tế là; thực sự; thật sự; thực tế")]},
    "規": {"viet": "QUY", "meaning_vi": "Cái khuôn tròn.", "meo": "", "reading": "キ", "vocab": [("党規", "とうき", "quy tắc Đảng"), ("内規", "ないき", "Nội qui riêng; qui định riêng")]},
    "係": {"viet": "HỆ", "meaning_vi": "Buộc, cũng nghĩa như chữ hệ [繫].", "meo": "", "reading": "かか.る かかり -がかり かか.わる", "vocab": [("係", "かかり", "sự chịu trách nhiệm"), ("係り", "かかり", "người phụ trách .")]},
    "折": {"viet": "CHIẾT, ĐỀ", "meaning_vi": "Gẫy, bẻ gẫy. Đỗ Mục [杜牧] : Chiết kích trầm sa thiết vị tiêu [折戟沉沙鐵未消] (Xích Bích hoài cổ [赤壁懷古]) Ngọn kích gẫy chìm trong bãi cát (đã lâu ngày) mà sắt vẫn chưa tiêu.", "meo": "", "reading": "お.る おり お.り -お.り お.れる", "vocab": [("折", "おり", "cơ hội; thời gian thích hợp; thời điểm thích hợp; dịp"), ("折々", "おりおり", "thỉnh thoảng")]},
    "液": {"viet": "DỊCH", "meaning_vi": "Nước dãi.", "meo": "", "reading": "エキ", "vocab": [("液", "えき", "dịch thể; dung dịch; dịch"), ("乳液", "にゅうえき", "Nhựa cây; mủ cây .")]},
    "類": {"viet": "LOẠI", "meaning_vi": "Loài giống. Như phân môn biệt loại [分門別類] chia từng môn ghẽ từng loài.", "meo": "", "reading": "たぐ.い", "vocab": [("類", "るい", "loại; chủng loại ."), ("類い", "るいい", "loài giống")]},
    "未": {"viet": "VỊ, MÙI", "meaning_vi": "Chi Vị, chi thứ tám trong 12 chi. Từ một giờ chiều đến ba giờ chiều gọi là giờ Vị. Thường quen đọc là chữ Mùi.", "meo": "", "reading": "いま.だ ま.だ ひつじ", "vocab": [("未", "ひつじ", "Mùi (dê); giờ Mùi"), ("未", "み", "vẫn chưa; chưa")]},
    "末": {"viet": "MẠT", "meaning_vi": "Ngọn. Như mộc mạt [木末] ngọn cây, trượng mạt [杖末] đầu gậy.", "meo": "", "reading": "すえ", "vocab": [("末", "うら", "đầu; cuối; đỉnh; chóp"), ("末", "すえ", "cuối; tương lai; đầu mút; chỗ tận cùng; hậu duệ; sau khi; rốt cục")]},
    "非": {"viet": "PHI", "meaning_vi": "Trái, không phải, sự vật gì có nghĩa nhất định, nếu không đúng hết đều gọi là phi.", "meo": "", "reading": "あら.ず", "vocab": [("非", "ひ", "phi; chẳng phải; trái"), ("非と", "ひと", "/'bændits/")]},
    "悲": {"viet": "BI", "meaning_vi": "Đau, khóc không có nước mắt gọi là bi. Đỗ Phủ [杜甫] : Vạn lý bi thu thường tác khách [萬里悲秋常作客] Ở xa muôn dặm, ta thường làm khách thương thu.", "meo": "", "reading": "かな.しい かな.しむ", "vocab": [("悲傷", "ひしょう", "bi thương"), ("悲劇", "ひげき", "bi kịch")]},
    "罪": {"viet": "TỘI", "meaning_vi": "Tội lỗi. Làm phạm phép luật phải phạt gọi là tội.", "meo": "", "reading": "つみ", "vocab": [("罪", "つみ", "tội ác; tội lỗi"), ("罪人", "ざいにん", "tội nhân")]},
    "息": {"viet": "TỨC", "meaning_vi": "Hơi thở. Mũi thở ra hít vào một lượt gọi là nhất tức [一息], thở dài mà than thở gọi là thái tức [太息].", "meo": "", "reading": "いき", "vocab": [("息", "いき", "hơi"), ("一息", "ひといき", "thổi phù; phụt ra từng luồng")]},
    "鼻": {"viet": "TỊ", "meaning_vi": "Cái mũi.", "meo": "", "reading": "はな", "vocab": [("鼻", "はな", "mũi ."), ("鼻下", "はなか", "sự tuyên dương")]},
    "各": {"viet": "CÁC", "meaning_vi": "Đều. Mỗi người có một địa vị riêng, không xâm lấn được. Như các bất tương mưu [各不相謀] đều chẳng cùng mưu.", "meo": "", "reading": "おのおの", "vocab": [("各", "かく", "mọi; mỗi"), ("各々", "かく々", "mỗi")]},
    "格": {"viet": "CÁCH, CÁC", "meaning_vi": "Chính. Như duy đại nhân vi năng cách quân tâm chi phi [惟大人爲能格君心之非] chỉ có bực đại nhân là chính được cái lòng xằng của vua.", "meo": "", "reading": "カク コウ キャク ゴウ", "vocab": [("格", "かく", "trạng thái; hạng"), ("人格", "じんかく", "nhân cách .")]},
    "落": {"viet": "LẠC", "meaning_vi": "Rơi, rụng, thất bại, bỏ lỡ", "meo": "Phần trên là bộ `艹` (thảo - cỏ cây). Phần dưới là `各` (các - mỗi). Hãy hình dung lá cây *rụng* (落) xuống *mỗi* (各) ngóc ngách trên cỏ (艹).", "reading": "おちる", "vocab": [("落ちる", "おちる", "rơi, rụng, thất bại"), ("落とす", "おとす", "đánh rơi, làm rớt")]},
    "絡": {"viet": "LẠC", "meaning_vi": "Quấn quanh, xe, quay. Như lạc ty [絡絲] quay tơ, nghĩa là quấn tơ vào cái vòng quay tơ, vì thế nên cái gì có ý ràng buộc đều gọi là lạc. Như lung lạc [籠絡], liên lạc [連絡], lạc dịch [絡繹] đều nói về ý nghĩa ràng buộc cả.", "meo": "", "reading": "から.む から.まる", "vocab": [("絡み", "からみ", "Sự kết nối; sự vướng mắc; sự liên can; mối quan hệ; liên quan; có liên quan"), ("絡む", "からむ", "cãi cọ")]},
    "路": {"viet": "LỘ", "meaning_vi": "Đường cái, đường đi lại. Như hàng lộ [航路] đường đi bể.", "meo": "", "reading": "-じ みち", "vocab": [("一路", "いちろ", "thẳng"), ("路上", "ろじょう", "trên con đường")]},
    "接": {"viet": "TIẾP", "meaning_vi": "Liền, hai đầu liền nhau gọi là tiếp.", "meo": "", "reading": "つ.ぐ", "vocab": [("接ぐ", "つぐ", "ghép (cây)"), ("交接", "こうせつ", "Sự giao hợp; giao hợp")]},
    "努": {"viet": "NỖ", "meaning_vi": "Gắng. Như nỗ lực [努力] gắng sức.", "meo": "", "reading": "つと.める", "vocab": [("努力", "どりょく", "chí tâm"), ("努めて", "つとめて", "làm việc chăm chỉ quá!")]},
    "怒": {"viet": "NỘ", "meaning_vi": "Giận. Cảm thấy một sự gì trái ý mà nổi cơn cáu tức lên gọi là chấn nộ [震怒] nghĩa là đùng đùng như sấm như sét, phần nhiều chỉ về sự giận của người tôn quý.", "meo": "", "reading": "いか.る おこ.る", "vocab": [("怒り", "いかり", "Cơn giận dữ; sự tức giận; sự nổi giận"), ("怒り", "おこり", "căm")]},
    "干": {"viet": "KIỀN, CAN, CÀN, CÁN", "meaning_vi": "Phạm. Như can phạm [干犯].", "meo": "", "reading": "ほ.す ほ.し- -ぼ.し ひ.る", "vocab": [("干", "ひ", "khô; sấy khô"), ("干す", "ほす", "phơi; làm khô")]},
    "汗": {"viet": "HÃN, HÀN", "meaning_vi": "Mồ hôi.", "meo": "", "reading": "あせ", "vocab": [("汗", "あせ", "mồ hôi"), ("冷汗", "ひやあせ", "mồ hôi lạnh")]},
    "岸": {"viet": "NGẠN", "meaning_vi": "Bờ. Như đê ngạn [堤岸] bờ đê. Tu đạo chứng chỗ cùng cực gọi là đạo ngạn [道岸] nghĩa là người hư hỏng nhờ có công tu học biết tới cõi hay, cũng như đắm đuối nhờ người cứu vớt vào tới bờ vậy. Trong kinh Phật nói tu chứng đến cõi chính giác là đáo bỉ ngạn [到彼岸], đăng giác ngạn [登覺岸] đều là cái nghĩa ấy cả.", "meo": "", "reading": "きし", "vocab": [("岸", "きし", "bờ"), ("傲岸", "ごうがん", "Tính kiêu kỳ")]},
    "平": {"viet": "BÌNH, BIỀN", "meaning_vi": "Bằng phẳng. Như thủy bình [水平] nước phẳng, địa bình [地平]đất phẳng. Hai bên cách nhau mà cùng tiến lên đều nhau gọi là bình hành tuyến [平行線].", "meo": "", "reading": "たい.ら -だいら ひら ひら-", "vocab": [("平", "ひら", "bằng"), ("平々", "たいら々", "ống bọt nước")]},
    "呼": {"viet": "HÔ, HÁ", "meaning_vi": "Thở ra. Đối lại với chữ hấp [吸].", "meo": "", "reading": "よ.ぶ", "vocab": [("呼ぶ", "よぶ", "gào"), ("呼値", "よびね", "giá chào bán .")]},
    "職": {"viet": "CHỨC", "meaning_vi": "Chức, phàm các việc quan đều gọi là chức. Như xứng chức [稱職] xứng đáng với cái chức của mình. Vì thế nên ngôi quan cũng gọi là chức. Như văn chức [文職] chức văn, vũ chức [武職] chức võ, v.v. Ngày xưa chư hầu vào chầu thiên tử xưng là thuật chức [述職] nghĩa là bày kể công việc của mình làm. Đời sau các quan ngoài vào chầu vua cũng xưng là thuật chức là vì đó.", "meo": "", "reading": "ショク ソク", "vocab": [("職", "しょく", "nghề nghiệp ."), ("下職", "したしょく", "người thầu phụ")]},
    "識": {"viet": "THỨC, CHÍ", "meaning_vi": "Biết, phân biệt, thấy mà nhận biết được.", "meo": "", "reading": "し.る しる.す", "vocab": [("識別", "しきべつ", "sự phân biệt"), ("卓識", "たくしき", "sự thâm nhập")]},
    "列": {"viet": "LIỆT", "meaning_vi": "Hàng lối, cái gì xếp một hàng thẳng gọi là hàng [行], xếp ngang gọi là liệt [列].", "meo": "", "reading": "レツ レ", "vocab": [("列", "れつ", "hàng; dãy"), ("一列", "いちれつ", "hàng")]},
    "例": {"viet": "LỆ", "meaning_vi": "Lệ, ví, lấy cái này làm mẫu mực cho cái kia gọi là lệ. Như thể lệ [體例], điều lệ [詞例], luật lệ [律例], v.v.", "meo": "", "reading": "たと.える", "vocab": [("例", "れい", "thí dụ"), ("例え", "たとえ", "ví dụ .")]},
    # "死" → already in main DB
    "績": {"viet": "TÍCH", "meaning_vi": "Đánh sợi, xe sợi. Nguyễn Du [阮攸] : Nữ sự duy tích ma [女事惟績麻] (Hoàng Mai sơn thượng thôn [黃梅山上村]) Việc đàn bà chỉ là xe sợi gai.", "meo": "", "reading": "セキ", "vocab": [("事績", "じせき", "thành tích"), ("功績", "こうせき", "công tích; công lao; công trạng; thành tích; thành tựu")]},
    "積": {"viet": "TÍCH, TÍ", "meaning_vi": "Chứa góp. Như tích trữ [積貯] cất chứa cho nhiều.", "meo": "", "reading": "つ.む -づ.み つ.もる つ.もり", "vocab": [("積む", "つむ", "chất; xếp"), ("体積", "たいせき", "thể tích .")]},
    "打": {"viet": "ĐẢ", "meaning_vi": "Đánh đập.", "meo": "", "reading": "う.つ う.ち- ぶ.つ", "vocab": [("打つ", "うつ", "bịch"), ("打つ", "ぶつ", "đánh")]},
    "貯": {"viet": "TRỮ", "meaning_vi": "Tích chứa. Như trữ tồn [貯存].", "meo": "", "reading": "た.める たくわ.える", "vocab": [("貯え", "たくわえ", "sự có nhiều"), ("貯える", "たくわえる", "bòn")]},
    "準": {"viet": "CHUẨN, CHUYẾT", "meaning_vi": "Bằng phẳng.", "meo": "", "reading": "じゅん.じる じゅん.ずる なぞら.える のり ひと.しい みずもり", "vocab": [("準", "じゅん", "chuẩn"), ("準備", "じゅんび", "sự chuẩn bị; sự sắp xếp; chuẩn bị; sắp xếp .")]},
    "進": {"viet": "TIẾN", "meaning_vi": "Tiến lên. Trái lại với chữ thoái [退].", "meo": "", "reading": "すす.む すす.める", "vocab": [("進み", "すすみ", "sự tiến tới"), ("進む", "すすむ", "tiến lên; tiến triển; tiến bộ")]},
    # "曜" → already in main DB
    "濯": {"viet": "TRẠC", "meaning_vi": "giặt, rửa", "meo": "Nước (氵) trắng (白) rộng (翟) dùng để giặt.", "reading": "たく", "vocab": [("洗濯", "せんたく", "giặt giũ"), ("洗濯機", "せんたくき", "máy giặt")]},
    "確": {"viet": "XÁC", "meaning_vi": "xác thực, chính xác, chắc chắn", "meo": "XÁC nhận điều gì đó bằng cách dùng ĐÁ (石) lưỡi CÂY (木) để đóng KHUNG (角) lại.", "reading": "かく", "vocab": [("確実", "かくじつ", "chắc chắn, xác thực"), ("確認", "かくにん", "xác nhận")]},
    "観": {"viet": "QUAN", "meaning_vi": "Quan sát,", "meo": "", "reading": "み.る しめ.す", "vocab": [("観", "かん", "bề ngoài; cảnh tượng; dáng vẻ"), ("主観", "しゅかん", "chủ quan; tưởng tượng chủ quan")]},
    "果": {"viet": "QUẢ", "meaning_vi": "Quả, trái cây. Như quả đào, quả mận, v.v.", "meo": "", "reading": "は.たす はた.す -は.たす は.てる -は.てる は.て", "vocab": [("果て", "はて", "sau cùng; cuối cùng; tận cùng ."), ("仏果", "ぶっか", "Niết bàn; nát bàn .")]},
    "単": {"viet": "ĐƠN", "meaning_vi": "Cô đơn, đơn độc, đơn chiếc", "meo": "", "reading": "ひとえ", "vocab": [("単", "たん", "đơn"), ("単に", "たんに", "một cách đơn thuần")]},
    "戦": {"viet": "CHIẾN", "meaning_vi": "Chiến tranh, chiến đấu", "meo": "", "reading": "いくさ たたか.う おのの.く そよ.ぐ わなな.く", "vocab": [("戦", "いくさ", "cuộc chiến tranh; trận chiến đấu; cuộc chiến; trận chiến; hiệp đấu"), ("戦", "せん", "chiến .")]},
    "任": {"viet": "NHÂM, NHẬM, NHIỆM", "meaning_vi": "Dốc lòng thành, lấy tâm ý cùng tin nhau gọi là nhâm.", "meo": "", "reading": "まか.せる まか.す", "vocab": [("任", "にん", "nhiệm vụ"), ("任す", "まかす", "dựa vào")]},
    "実": {"viet": "THỰC", "meaning_vi": "Sự thực, chân thực", "meo": "", "reading": "み みの.る まこと みの みち.る", "vocab": [("実", "じつ", "sự chân thực; sự chân thành; sự thành thực"), ("実", "み", "quả")]},
    # "業" → already in main DB
    "残": {"viet": "TÀN", "meaning_vi": "Giản thể của chữ [殘].", "meo": "", "reading": "のこ.る のこ.す そこな.う のこ.り", "vocab": [("残す", "のこす", "bám chặt (Sumô)"), ("残り", "のこり", "cái còn lại; phần còn lại; phần rơi rớt lại")]},
    "浅": {"viet": "THIỂN, TIÊN", "meaning_vi": "Giản thể của chữ [淺].", "meo": "", "reading": "あさ.い", "vocab": [("浅い", "あさい", "cạn"), ("浅学", "せんがく", "hiểu biết nông cạn; thiển cận")]},
    "消": {"viet": "TIÊU", "meaning_vi": "Mất đi, hết.", "meo": "", "reading": "き.える け.す", "vocab": [("消す", "けす", "bôi"), ("消光", "しょうこう", "thời gian yên tựnh")]},
    "具": {"viet": "CỤ", "meaning_vi": "Bày đủ. Như cụ thực [具食] bày biện đủ các đồ ăn.", "meo": "", "reading": "そな.える つぶさ.に", "vocab": [("具", "ぐ", "dụng cụ"), ("具に", "つぶさに", "hoàn toàn")]},
    "昔": {"viet": "TÍCH", "meaning_vi": "Xưa, trước. Như tích nhật [昔日] ngày xưa. Nguyễn Trãi [阮廌] : Loạn hậu phùng nhân phi túc tích [亂後逢人非夙昔] (Thu dạ khách cảm [秋夜客感]) Sau cơn ly loạn, người gặp không ai là kẻ quen biết cũ.", "meo": "", "reading": "むかし", "vocab": [("昔", "むかし", "cổ"), ("昔々", "むかしむかし", "ngày xửa ngày xưa .")]},
    # "借" → already in main DB
    "兆": {"viet": "TRIỆU", "meaning_vi": "Điềm, đời xưa dùng mai rùa bói, đốt mai rùa, rồi coi những đường nứt mà đoán tốt xấu gọi là triệu. Phàm dùng cái gì để xem tốt xấu đều gọi là triệu. Như cát triệu [吉兆] điềm tốt.", "meo": "", "reading": "きざ.す きざ.し", "vocab": [("兆", "きざし", "triệu chứng; điềm báo; dấu hiệu"), ("兆", "ちょう", "nghìn tỷ .")]},
    "包": {"viet": "BAO", "meaning_vi": "Bọc, dùng đồ bọc ngoài cái gì gọi là bao.", "meo": "", "reading": "つつ.む くる.む", "vocab": [("包み", "つつみ", "gói; bọc"), ("包む", "くるむ", "bọc; gói; bao bọc")]},
    "補": {"viet": "BỔ", "meaning_vi": "Vá áo.", "meo": "", "reading": "おぎな.う", "vocab": [("補い", "おぎない", "sự làm đầy"), ("補う", "おぎなう", "đền bù; bù; bổ sung")]},
    "議": {"viet": "NGHỊ", "meaning_vi": "Bàn, bàn về sự lý để phân biệt phải trái gọi là luận [論], bàn về sự lý để định việc nên hay không gọi là nghị [議]. Như hội nghị [會議] họp bàn, quyết nghị [決議] bàn cho quyết xong để thi hành.", "meo": "", "reading": "ギ", "vocab": [("争議", "そうぎ", "sự bãi công; cuộc bãi công"), ("議事", "ぎじ", "nghị sự")]},
    "他": {"viet": "THA", "meaning_vi": "Khác, là kẻ kia. Như tha nhân [他人] người khác, tha sự [他事] việc khác, v.v.", "meo": "", "reading": "ほか", "vocab": [("他", "ほか", "ngoài ."), ("他の", "ほかの", "khác .")]},
    "的": {"viet": "ĐÍCH, ĐỂ", "meaning_vi": "Thấy rõ, lộ ra ngoài. Như tiểu nhân chi đạo, đích nhiên nhi nhật vong [小人之道，的然而日亡] (Lễ Ký [禮記]) đạo kẻ tiểu nhân bề ngoài rõ vậy mà ngày mất dần đi.", "meo": "", "reading": "まと", "vocab": [("的", "てき", "đích"), ("的", "まと", "cái đích")]},
    "約": {"viet": "ƯỚC", "meaning_vi": "Thắt, bó. Như ước phát [約髮] búi tóc, ước túc [約足] bó chân.", "meo": "", "reading": "つづ.まる つづ.める つづま.やか", "vocab": [("約", "やく", "chừng"), ("予約", "よやく", "dự ước")]},
    "均": {"viet": "QUÂN, VẬN", "meaning_vi": "Đều, không ai hơn kém gọi là quân.", "meo": "", "reading": "なら.す", "vocab": [("均し", "ひとし", "số trung bình"), ("均一", "きんいつ", "toàn bộ như nhau; quân nhất; toàn bộ đều nhau; giống nhau; đồng đều; thống nhất")]},
    "談": {"viet": "ĐÀM", "meaning_vi": "Bàn bạc, hai bên cùng nhau bàn bạc sự vật lung tung đều gọi là đàm. Như thanh đàm [清談] bàn suông.", "meo": "", "reading": "ダン", "vocab": [("会談", "かいだん", "hội đàm"), ("余談", "よだん", "sự nói ngoài lề")]},
    "再": {"viet": "TÁI", "meaning_vi": "Hai, lại. Như tái tam [再三] luôn mãi, tái phạm [再犯] lại phạm lần nữa, tái tiếu [再醮] đàn bà lấy chồng lần thứ hai.", "meo": "", "reading": "ふたた.び", "vocab": [("再", "さい", "tái; lại một lần nữa"), ("再々", "さいさい", "thường")]},
    "構": {"viet": "CẤU", "meaning_vi": "Dựng nhà. Con nối nghiệp cha gọi là khẳng đường khẳng cấu [肯堂肯構].", "meo": "", "reading": "かま.える かま.う", "vocab": [("構う", "かまう", "chăm sóc; săn sóc"), ("構え", "かまえ", "tư thế; dáng điệu")]},
    "婚": {"viet": "HÔN", "meaning_vi": "Lấy vợ, con dâu.", "meo": "", "reading": "コン", "vocab": [("婚儀", "こんぎ", "Nghi lễ hôn lễ; nghi lễ kết hôn ."), ("再婚", "さいこん", "cải giá")]},
    "底": {"viet": "ĐỂ", "meaning_vi": "Đáy. Như thủy để [水底] đáy nước. Nguyễn Du [阮攸] : Nhãn để phù vân khan thế sự [眼底浮雲看世事] (Ký hữu [寄友]) Xem việc đời như mây nổi trong đáy mắt.", "meo": "", "reading": "そこ", "vocab": [("底", "そこ", "đáy"), ("底値", "そこね", "giá sàn")]},
    "愛": {"viet": "ÁI", "meaning_vi": "Yêu thích. Như ái mộ [愛慕] yêu mến.", "meo": "", "reading": "いと.しい かな.しい め.でる お.しむ まな", "vocab": [("愛", "あい", "tình yêu; tình cảm"), ("愛す", "あいす", "chuộng .")]},
    "酔": {"viet": "TÚY", "meaning_vi": "Say", "meo": "", "reading": "よ.う よ.い よ", "vocab": [("酔い", "よい", "say"), ("酔う", "よう", "say rượu")]},
    "季": {"viet": "QUÝ", "meaning_vi": "Nhỏ. Em bé gọi là quý đệ [季弟].", "meo": "", "reading": "キ", "vocab": [("季", "き", "mùa"), ("乾季", "かんき", "mùa khô")]},
    "難": {"viet": "NAN, NẠN", "meaning_vi": "Khó, trái lại với dị [易] dễ.", "meo": "", "reading": "かた.い -がた.い むずか.しい むづか.しい むつか.しい -にく.い", "vocab": [("難", "なん", "sự khó khăn"), ("難い", "かたい", "khó; khó khăn")]},
    "予": {"viet": "DỰ", "meaning_vi": "trước, dự định", "meo": "Mái nhà (冖) có móc câu (亅) báo hiệu trước", "reading": "よ", "vocab": [("予定", "よてい", "dự định"), ("予習", "よしゅう", "soạn bài")]},
    "橋": {"viet": "KIỀU, KHIÊU, CAO", "meaning_vi": "Cái cầu (cầu cao mà cong).", "meo": "", "reading": "はし", "vocab": [("橋", "はし", "cầu"), ("橋台", "きょうだい", "mố cầu .")]},
    "布": {"viet": "BỐ", "meaning_vi": "Vải, những đồ dệt bằng gai bằng sợi bông gọi là bố.", "meo": "", "reading": "ぬの", "vocab": [("布", "ぬの", "vải"), ("布令", "ふれい", "sự công bố")]},
    "希": {"viet": "HI", "meaning_vi": "Ít. Như ki hi [幾希] hầu ít, hiếm, hi hãn [希罕] hiếm có, hy kì [希奇] lạ lùng ít thấy, v.v.", "meo": "", "reading": "まれ", "vocab": [("希", "まれ", "hiếm"), ("希世", "きせい", "hiếm")]},
    "定": {"viet": "ĐỊNH, ĐÍNH", "meaning_vi": "Định, xếp đặt được yên ổn, không bị lay động nữa gọi là định. Nguyễn Du [阮攸] : Đình vân xứ xứ tăng miên định [停雲處處僧眠定] (Vọng quan âm miếu [望觀音廟]) Mây ngưng chốn chốn sư yên giấc.", "meo": "", "reading": "さだ.める さだ.まる さだ.か", "vocab": [("定か", "さだか", "rõ ràng; phân minh"), ("定め", "さだめ", "phép")]},
    "迎": {"viet": "NGHÊNH, NGHỊNH", "meaning_vi": "Đón. Chờ vật ngoài nó tới mà ngửa mặt ra đón lấy gọi là nghênh. Như tống nghênh [送迎] đưa đón. Hoan nghênh [歡迎] vui đón. Nghênh phong [迎風] hóng gió, v.v.", "meo": "", "reading": "むか.える", "vocab": [("迎え", "むかえ", "việc tiếp đón; người tiếp đón"), ("迎合", "げいごう", "sự nắm được ý người khác; sự đón được suy nghĩ của người khác; sự tâng bốc; sự xu nịnh")]},
    "卵": {"viet": "NOÃN", "meaning_vi": "Cái trứng. Như nguy như lũy noãn [危如累卵] nguy như trứng xếp chồng, thế như noãn thạch [勢如卵石] thế như trứng với đá. Nghĩa là cứng mềm không chịu nổi nhau vậy. Nuôi nấng cũng gọi là noãn dực [卵翼] nghĩa là như chim ấp trứng vậy.", "meo": "", "reading": "たまご", "vocab": [("卵", "たまご", "trứng; quả trứng"), ("卵円", "たまごえん", "có hình trái xoan")]},
    "徒": {"viet": "ĐỒ", "meaning_vi": "Đi bộ. Lính bộ binh cũng gọi là đồ. Như công đồ tam vạn [公徒三萬] bộ binh nhà vua tam vạn. Xe của vua đi cũng gọi là đồ. Như đồ ngự bất kinh [徒御不驚] xe vua chẳng sợ.", "meo": "", "reading": "いたずら あだ", "vocab": [("徒", "と", "vô hiệu"), ("仏徒", "ぶっと", "Tín đồ phật giáo .")]},
    "乳": {"viet": "NHŨ", "meaning_vi": "Cái vú, các loài động vật đều có vú để cho con bú.", "meo": "", "reading": "ちち ち", "vocab": [("乳", "ちち", "nhũ"), ("乳価", "にゅうか", "thể sữa")]},
    "礼": {"viet": "LỄ", "meaning_vi": "Cũng như chữ lễ [禮].", "meo": "", "reading": "レイ ライ", "vocab": [("礼", "れい", "sự biểu lộ lòng biết ơn ."), ("お礼", "おれい", "cám ơn")]},
    "示": {"viet": "KÌ, THỊ", "meaning_vi": "Thần đất, cùng nghĩa với chữ kì [祇].", "meo": "", "reading": "しめ.す", "vocab": [("示し", "しめし", "kỷ luật"), ("示す", "しめす", "biểu hiện ra; chỉ ra; cho thấy")]},
    "表": {"viet": "BIỂU", "meaning_vi": "Cái áo ngoài.", "meo": "", "reading": "おもて -おもて あらわ.す あらわ.れる あら.わす", "vocab": [("表", "おもて", "bề phải"), ("表", "ひょう", "biểu; bảng; bảng biểu")]},
    "倍": {"viet": "BỘI", "meaning_vi": "Gấp nhiều lần. Như bội nhị [倍二] gấp hai, bội tam [倍三] gấp ba, v.v.", "meo": "", "reading": "バイ", "vocab": [("倍", "ばい", "sự gấp đôi ."), ("一倍", "いちばい", "một phần; gấp đôi")]},
    "対": {"viet": "ĐỐI", "meaning_vi": "Cũng như chữ đối [對].", "meo": "", "reading": "あいて こた.える そろ.い つれあ.い なら.ぶ むか.う", "vocab": [("対", "たい", "đối"), ("対", "つい", "sự đối")]},
    "付": {"viet": "PHÓ", "meaning_vi": "Giao phó cho.", "meo": "", "reading": "つ.ける -つ.ける -づ.ける つ.け つ.け- -つ.け -づ.け -づけ つ.く -づ.く つ.き -つ.き -つき -づ.き -づき", "vocab": [("付", "つけ", "sự thêm vào; phần thêm vào"), ("付き", "つき", "ấn tượng")]},
    "守": {"viet": "THỦ, THÚ", "meaning_vi": "Giữ, coi. Như bảo thủ [保守] ôm giữ.", "meo": "", "reading": "まも.る まも.り もり -もり かみ", "vocab": [("守", "もり", "bảo mẫu; người trông trẻ"), ("守り", "まもり", "thủ .")]},
    "顔": {"viet": "NHAN", "meaning_vi": "Dị dạng của chữ [颜].", "meo": "", "reading": "かお", "vocab": [("顔", "かお", "diện mạo"), ("似顔", "にがお", "chân dung")]},
    "由": {"viet": "DO", "meaning_vi": "Bởi, tự.", "meo": "", "reading": "よし よ.る", "vocab": [("由", "よし", "lý do; nguyên nhân"), ("事由", "じゆう", "nguyên nhân .")]},
    "演": {"viet": "DIỄN", "meaning_vi": "Diễn ra. Sự gì nhân cái này được cái kia, có thể y theo cái lẽ tự nhiên mà suy ra đều gọi là diễn. Như nhân tám quẻ (bát quái [八卦]) mà diễn ra 64 quẻ, gọi là diễn dịch [演易].", "meo": "", "reading": "エン", "vocab": [("上演", "じょうえん", "bản tưồng"), ("主演", "しゅえん", "vai diễn .")]},
    "横": {"viet": "HOÀNH, HOẠNH, QUÁNG", "meaning_vi": "Giản thể của chữ [橫].", "meo": "", "reading": "よこ", "vocab": [("横", "よこ", "bề ngang"), ("横に", "よこに", "qua")]},
    "率": {"viet": "SUẤT, SÚY, LUẬT, SOÁT", "meaning_vi": "Noi theo. Nguyễn Du [阮攸] : Hồn hề ! hồn hề ! suất thử đạo [魂兮魂兮率此道] (Phản chiêu hồn [反招魂]) Hồn ơi ! hồn ơi ! nếu cứ noi theo lối đó.", "meo": "", "reading": "ひき.いる", "vocab": [("率", "りつ", "hệ số [vật lý]; tỷ lệ"), ("低率", "ていりつ", "tỷ lệ thấp")]},
    "老": {"viet": "LÃO", "meaning_vi": "Người già bảy mươi tuổi. Phàm người nào có tuổi tác đều gọi là lão.", "meo": "", "reading": "お.いる ふ.ける", "vocab": [("老", "ろう", "tuổi già"), ("老い", "おい", "tuổi già")]},
    # "悪" → already in main DB
    "要": {"viet": "YẾU, YÊU", "meaning_vi": "Thiết yếu, đúng sự lý gọi là yếu. Như yếu nghĩa [要義] nghĩa thiết yếu, đề yếu [提要] nhắc cái chỗ thiết yếu lên.", "meo": "", "reading": "い.る かなめ", "vocab": [("要", "かなめ", "điểm thiết yếu; điểm chủ yếu; điểm chủ chốt"), ("要り", "いり", "những người nghe")]},
    "独": {"viet": "ĐỘC", "meaning_vi": "Tục dùng như chữ độc [獨].", "meo": "", "reading": "ひと.り", "vocab": [("独", "どいつ", "nước Đức"), ("独", "どく", "độc .")]},
    "困": {"viet": "KHỐN", "meaning_vi": "Khốn cùng. Phàm các sự nhọc mệt quẫn bách đều gọi là khốn.", "meo": "", "reading": "こま.る", "vocab": [("困る", "こまる", "bối rối"), ("困却", "こんきゃく", "sự lúng túng")]},
    "団": {"viet": "ĐOÀN", "meaning_vi": "Một dạng của chữ đoàn [團].", "meo": "", "reading": "かたまり まる.い", "vocab": [("団", "だん", "toán ."), ("一団", "いちだん", "thân thể")]},
    "冊": {"viet": "SÁCH", "meaning_vi": "Một dạng của chữ [册].", "meo": "", "reading": "ふみ", "vocab": [("冊", "さつ", "tập"), ("一冊", "いちさつ", "văn kiện; tài liệu")]},
    "編": {"viet": "BIÊN", "meaning_vi": "Cái lề sách. Như Khổng Tử độc Dịch, vi biên tam tuyệt [孔子讀易, 韋編三絕] (Hán thư [漢書]) đức Khổng Tử đọc Kinh Dịch ba lần đứt lề sách.", "meo": "", "reading": "あ.む -あ.み", "vocab": [("編", "へん", "sự biên soạn"), ("編む", "あむ", "bện")]},
    "亡": {"viet": "VONG, VÔ", "meaning_vi": "Mất. Như Lương vong [梁亡] nước Lương mất rồi.", "meo": "", "reading": "な.い な.き- ほろ.びる ほろ.ぶ ほろ.ぼす", "vocab": [("亡い", "ない", "chết"), ("亡児", "ぼうじ", "giới tính")]},
    "忘": {"viet": "VONG", "meaning_vi": "Quên. Nguyễn Trãi [阮廌] : Nhật trường ẩn kỷ vong ngôn xứ [日長隱几忘言處] (Đề Trình xử sĩ Vân oa đồ [題程處士雲窩圖]) Ngày dài tựa ghế, quên nói năng.", "meo": "", "reading": "わす.れる", "vocab": [("健忘", "けんぼう", "sự đãng trí; chứng đãng trí; chứng quên; đãng trí; hay quên; tật hay quên"), ("備忘", "びぼう", "cái nhắc nhở")]},
    "望": {"viet": "VỌNG", "meaning_vi": "Trông xa. Như chiêm vọng [瞻望] trông mong.", "meo": "", "reading": "のぞ.む もち", "vocab": [("望み", "のぞみ", "sự trông mong; nguyện vọng"), ("望む", "のぞむ", "nguyện")]},
    "曲": {"viet": "KHÚC", "meaning_vi": "Cong, lẽ không được thẳng cứng gọi là khúc.", "meo": "", "reading": "ま.がる ま.げる くま", "vocab": [("曲", "きょく", "khúc; từ (ca nhạc)"), ("曲げ", "まげ", "sự uốn cong")]},
    "農": {"viet": "NÔNG", "meaning_vi": "Nghề làm ruộng.", "meo": "", "reading": "ノウ", "vocab": [("農", "のう", "nông nghiệp"), ("中農", "ちゅうのう", "trung nông .")]},
    "豊": {"viet": "PHONG", "meaning_vi": "phong phú.", "meo": "", "reading": "ゆた.か とよ", "vocab": [("豊か", "ゆたか", "phong phú; dư dật; giàu có"), ("豊作", "ほうさく", "mùa màng bội thu")]},
    # "以" → already in main DB
    "期": {"viet": "KÌ, KI", "meaning_vi": "Kì hẹn. Như khiên kì [愆期] sai hẹn.", "meo": "", "reading": "キ ゴ", "vocab": [("期", "き", "kì; thời gian"), ("期す", "きす", "mong chờ")]},
    "欠": {"viet": "KHIẾM", "meaning_vi": "Ngáp. Như khiếm thân [欠伸] vươn vai ngáp.", "meo": "", "reading": "か.ける か.く", "vocab": [("欠", "けつ", "sự thiếu"), ("欠く", "かく", "thiếu")]},
    "次": {"viet": "THỨ", "meaning_vi": "Lần lượt, dưới bậc trên trở xuống bét đều gọi là thứ.", "meo": "", "reading": "つ.ぐ つぎ", "vocab": [("次", "つぎ", "lần sau; sau đây; tiếp đến"), ("次々", "つぎつぎ", "lần lượt kế tiếp .")]},
    "組": {"viet": "TỔ", "meaning_vi": "Dây thao, đời xưa dùng dây thao để đeo ấn, cho nên gọi người bỏ chức quan về là giải tổ [解組].", "meo": "", "reading": "く.む くみ -ぐみ", "vocab": [("組", "くみ", "bộ"), ("組み", "くみ", "sự hợp thành")]},
    "助": {"viet": "TRỢ", "meaning_vi": "Giúp. Mượn sức cái này giúp thêm cái kia.", "meo": "", "reading": "たす.ける たす.かる す.ける すけ", "vocab": [("助", "すけ", "sự giúp đỡ"), ("助け", "たすけ", "sự giúp đỡ")]},
    "査": {"viet": "TRA", "meaning_vi": "kiểm tra, xem xét", "meo": "Cây (木) tre (一) làm nhà (⻖) phải tra xét kỹ.", "reading": "さ", "vocab": [("検査", "けんさ", "kiểm tra"), ("査察", "ささつ", "thăm dò")]},
    "商": {"viet": "THƯƠNG", "meaning_vi": "Đắn đo. Như thương lượng [商量], thương chước [商酌] nghĩa là bàn bạc, đắn đo với nhau.", "meo": "", "reading": "あきな.う", "vocab": [("商", "しょう", "số thương"), ("商い", "あきない", "nghề")]},
    "欧": {"viet": "ÂU, ẨU", "meaning_vi": "Giản thể của chữ 歐", "meo": "", "reading": "うた.う は.く", "vocab": [("欧化", "おうか", "sự âu hoá"), ("南欧", "なんおう", "Nam Âu")]},
    "復": {"viet": "PHỤC, PHÚC", "meaning_vi": "Lại. Đã đi rồi trở lại gọi là phục.", "meo": "", "reading": "また", "vocab": [("復仇", "ふくきゅう", "Sự trả thù; sự báo thù ."), ("復位", "ふくい", "phục vị .")]},
    "複": {"viet": "PHỨC", "meaning_vi": "Áo kép.", "meo": "", "reading": "フク", "vocab": [("複", "ふく", "đôi"), ("複写", "ふくしゃ", "bản sao; bản in lại")]},
    "取": {"viet": "THỦ", "meaning_vi": "cầm, lấy, nhận", "meo": "Tay (又) cầm lấy (耳) cái tai (取) của con vật.", "reading": "と.る", "vocab": [("取得", "しゅとく", "thu được, giành được"), ("取材", "しゅざい", "thu thập tài liệu, phỏng vấn")]},
    "最": {"viet": "TỐI", "meaning_vi": "Rất. Như tối hảo [最好] rất tốt.", "meo": "", "reading": "もっと.も つま", "vocab": [("最も", "もっとも", "vô cùng; cực kỳ; cực độ"), ("最上", "さいじょう", "sự tối thượng; sự tốt nhất; sự cao nhất")]},
    "棒": {"viet": "BỔNG", "meaning_vi": "Cái gậy.", "meo": "", "reading": "ボウ", "vocab": [("棒", "ぼう", "gậy"), ("乳棒", "にゅうぼう", "Cái chày .")]},
    "祭": {"viet": "TẾ, SÁI", "meaning_vi": "Cúng tế.", "meo": "", "reading": "まつ.る まつ.り まつり", "vocab": [("祭", "まつり", "thuộc ngày hội"), ("祭り", "まつり", "hội hè .")]},
    "悩": {"viet": "NÃO", "meaning_vi": "Khổ não, lo lắng", "meo": "", "reading": "なや.む なや.ます なや.ましい なやみ", "vocab": [("悩み", "なやみ", "bệnh tật"), ("悩む", "なやむ", "khổ đau; lo lắng; buồn phiền")]},
    "勝": {"viet": "THẮNG, THĂNG", "meaning_vi": "Được, đánh được quân giặc gọi là thắng. Như bách chiến bách thắng [百戰百勝] trăm trận đánh được cả trăm.", "meo": "", "reading": "か.つ -が.ち まさ.る すぐ.れる かつ", "vocab": [("勝ち", "かち", "chiến thắng"), ("勝つ", "かつ", "được")]},
    "階": {"viet": "GIAI", "meaning_vi": "Bực thềm, thềm cao hơn sàn, phải xây bực lên gọi là giai. Cao Bá Quát [高伯适] : Tiền giai yêu khách chỉ [前階要客止] (Phạn xá cảm tác [飯舍感作]) Trước thềm kèo nài khách dừng chân.", "meo": "", "reading": "きざはし", "vocab": [("階", "かい", "lầu"), ("一階", "いちかい", "tầng hai")]},
    "労": {"viet": "LAO", "meaning_vi": "Lao động, lao lực, công lao", "meo": "", "reading": "ろう.する いたわ.る いた.ずき ねぎら つか.れる ねぎら.う", "vocab": [("労", "ろう", "sự lao động; sự khó nhọc"), ("労り", "いたわり", "điều lo lắng")]},
    "豆": {"viet": "ĐẬU", "meaning_vi": "Bát đậu, cái bát tiện bằng gỗ để đựng phẩm vật cúng hoặc các thức dưa, giấm v.v. Tự thiên dụng ngõa đậu [祀天用瓦豆] tế trời dùng bát bằng đất nung.", "meo": "", "reading": "まめ まめ-", "vocab": [("豆", "まめ", "đậu ."), ("豆乳", "とうにゅう", "sữa đậu nành .")]},
    "頭": {"viet": "ĐẦU", "meaning_vi": "Bộ đầu, đầu lâu. Như nhân đầu [人頭] đầu người, ngưu đầu [牛頭] đầu bò.", "meo": "", "reading": "あたま かしら -がしら かぶり", "vocab": [("頭", "あたま", "cái đầu"), ("頭", "かしら", "đầu")]},
    "登": {"viet": "ĐĂNG", "meaning_vi": "Lên. Như đăng lâu [登樓] lên lầu.", "meo": "", "reading": "のぼ.る あ.がる", "vocab": [("登り", "のぼり", "sự trèo lên"), ("登る", "のぼる", "được đưa ra; được đặt ra (trong chương trình)")]},
    "喜": {"viet": "HỈ, HÍ, HI", "meaning_vi": "Mừng. Như hoan hỉ [歡喜] vui mừng.", "meo": "", "reading": "よろこ.ぶ よろこ.ばす", "vocab": [("喜び", "よろこび", "hân hạnh"), ("喜ぶ", "よろこぶ", "hí hửng")]},
    "突": {"viet": "ĐỘT", "meaning_vi": "Chợt, thốt nhiên. Thốt nhiên mà đến gọi là đột như kỳ lai [突如其來] (Dịch Kinh [易經], quẻ Li).", "meo": "", "reading": "つ.く", "vocab": [("突き", "つき", "sự đẩy mạnh"), ("突く", "つつく", "chống")]},
    "涙": {"viet": "LỆ", "meaning_vi": "Nước mắt", "meo": "", "reading": "なみだ", "vocab": [("涙", "なみだ", "châu lệ"), ("催涙", "さいるい", "nước mắt")]},
    "機": {"viet": "KI, CƠ", "meaning_vi": "Cái nẫy, cái máy, phàm cái gì do đấy mà phát động ra đều gọi là ki. Như ki quan [機關], sự ki [事機], ki trữ [機杼] cái máy dệt, cái khung cửi.", "meo": "", "reading": "はた", "vocab": [("機", "はた", "máy dệt ."), ("機上", "きじょう", "lý thuyết; có tính chất lý thuyết")]},
    # "暖" → already in main DB
    "机": {"viet": "KY, CƠ", "meaning_vi": "Giản thể của chữ [機].", "meo": "", "reading": "つくえ", "vocab": [("机", "つくえ", "bàn"), ("机上", "きじょう", "trên bàn; lý thuyết; trên giấy tờ")]},
    "築": {"viet": "TRÚC", "meaning_vi": "Đắp đất, lèn đất. Xây đắp cái gì cũng phải lên cái nền cho tốt đã, cho nên các việc xây đắp nhà cửa đều gọi là kiến trúc [建築].", "meo": "", "reading": "きず.く", "vocab": [("築く", "きずく", "xây dựng"), ("修築", "しゅうちく", "sự sửa chữa")]},
    "枚": {"viet": "MAI", "meaning_vi": "Cái quả, gốc cây. Như tảo nhất mai [棗一枚] một quả táo. Mai bốc công thần [枚卜功臣] nhất nhất đều bói xem ai công hơn, đời sau gọi sự dùng quan tể tướng là mai bốc [枚卜] là bởi đó.", "meo": "", "reading": "マイ バイ", "vocab": [("枚", "まい", "tấm; tờ"), ("三枚", "さんまい", "kịch vui")]},
    "燃": {"viet": "NHIÊN", "meaning_vi": "Đốt. Pháp Hoa Kinh [法華經] : Nhiên hương du tô đăng [燃香油酥燈] (Phân biệt công đức phẩm [分別功德品]) Đốt đèn dầu nến thơm.", "meo": "", "reading": "も.える も.やす も.す", "vocab": [("燃す", "もす", "đốt"), ("燃す", "もやす", "đốt; thổi bùng")]},
    "易": {"viet": "DỊCH, DỊ", "meaning_vi": "Đổi. Hai bên lấy tiền hay lấy đồ mà đổi cho nhau gọi là mậu dịch [貿易]. Dịch tử nhi giáo [易子而教] Đổi con cho nhau mà dạy. Ngày xưa thường dùng cách ấy, vì mình dạy con mình thường không nghiêm bằng người khác.", "meo": "", "reading": "やさ.しい やす.い", "vocab": [("易", "えき", "sự đoán"), ("易い", "やすい", "dễ; dễ dàng")]},
    "探": {"viet": "THAM, THÁM", "meaning_vi": "Tìm tòi.", "meo": "", "reading": "さぐ.る さが.す", "vocab": [("探す", "さがす", "kiếm"), ("探り", "さぐり", "nghe kêu")]},
    "深": {"viet": "THÂM", "meaning_vi": "Bề sâu. Như thâm nhược can xích [深若干尺] sâu ngần ấy thước.", "meo": "", "reading": "ふか.い -ぶか.い ふか.まる ふか.める み-", "vocab": [("深い", "ふかい", "dày"), ("深さ", "ふかさ", "bề sâu")]},
    "禁": {"viet": "CẤM, CÂM", "meaning_vi": "Cấm chế.", "meo": "", "reading": "キン", "vocab": [("禁", "きん", "sự cấm"), ("禁中", "きんちゅう", "sân nhà")]},
    "賛": {"viet": "TÁN", "meaning_vi": "Tục dùng như chữ tán [贊].", "meo": "", "reading": "たす.ける たた.える", "vocab": [("賛", "さん", "truyện cổ tích"), ("賛助", "さんじょ", "sự trợ giúp")]},
    "葉": {"viet": "DIỆP, DIẾP", "meaning_vi": "Lá, lá cây cỏ, cánh hoa. Như trúc diệp [竹葉] lá tre, thiên diệp liên [千葉蓮] hoa sen nghìn cánh.", "meo": "", "reading": "は", "vocab": [("葉", "は", "diệp"), ("一葉", "いちよう", "cây đuôi chồn; một chiếc lá")]},
    "倒": {"viet": "ĐẢO", "meaning_vi": "Ngã.", "meo": "", "reading": "たお.れる -だお.れ たお.す さかさま さかさ さかしま", "vocab": [("倒さ", "たおさ", "bị nghịch đảo"), ("倒す", "たおす", "chặt đổ; đốn; lật đổ; quật ngã; giết; làm ngã; đánh gục")]},
    # "寝" → already in main DB
    "疑": {"viet": "NGHI", "meaning_vi": "Ngờ, lòng chưa tin đích gọi là nghi.", "meo": "", "reading": "うたが.う", "vocab": [("疑", "うたぐ", "sự nghi ngờ"), ("疑い", "うたがい", "sự nghi ngờ")]},
    "術": {"viet": "THUẬT", "meaning_vi": "Nghề thuật. Kẻ có nghề riêng đi các nơi kiếm tiền gọi là thuật sĩ [術士].", "meo": "", "reading": "すべ", "vocab": [("術", "じゅつ", "kĩ nghệ; kĩ thuật; đối sách; kế sách ."), ("術中", "じゅっちゅう", "mưu mẹo")]},
    "州": {"viet": "CHÂU", "meaning_vi": "Châu, ngày xưa nhân thấy có núi cao sông dài mới chia đất ra từng khu lấy núi sông làm mốc nên gọi là châu.", "meo": "", "reading": "す", "vocab": [("州", "しゅう", "tỉnh; nhà nước"), ("州", "す", "bãi cát (ở biển) .")]},
    "順": {"viet": "THUẬN", "meaning_vi": "Theo, bé nghe lớn chỉ bảo không dám trái một tí gì gọi là thuận. Như thuận tòng [順從] tuân theo.", "meo": "", "reading": "ジュン", "vocab": [("順", "じゅん", "trật tự; lượt ."), ("abc順", "エービーシーじゅん", "thứ tự abc .")]},
    "流": {"viet": "LƯU", "meaning_vi": "Nước chảy. Như hải lưu [海流] dòng nuớc biển.", "meo": "", "reading": "なが.れる なが.れ なが.す -なが.す", "vocab": [("流", "りゅう", "dòng; phong cách; tính chất"), ("流し", "ながし", "bồn rửa; chậu rửa")]},
    "焼": {"viet": "THIÊU", "meaning_vi": "Thiêu đốt", "meo": "", "reading": "や.く や.き や.き- -や.き や.ける", "vocab": [("焼く", "やく", "đốt cháy"), ("全焼", "ぜんしょう", "sự thiêu trụi hoàn toàn; sự phá hủy hoàn toàn")]},
    "束": {"viet": "THÚC, THÚ", "meaning_vi": "Buộc, bó lại. Như thúc thủ [束手] bó tay.", "meo": "", "reading": "たば たば.ねる つか つか.ねる", "vocab": [("束", "たば", "bó; búi; cuộn"), ("束ね", "たばね", "bó")]},
    "恋": {"viet": "LUYẾN", "meaning_vi": "Giản thể của chữ [戀].", "meo": "", "reading": "こ.う こい こい.しい", "vocab": [("恋", "こい", "tình yêu"), ("恋う", "こう", "yêu .")]},
    "変": {"viet": "BIẾN", "meaning_vi": "Biến đổi, biến thiên", "meo": "", "reading": "か.わる か.わり か.える", "vocab": [("変", "へん", "dấu giáng (âm nhạc)"), ("変え", "かえ", "hay thay đổi")]},
    "訳": {"viet": "DỊCH", "meaning_vi": "Phiên dịch.", "meo": "", "reading": "わけ", "vocab": [("訳", "わけ", "lý do; nguyên nhân"), ("訳す", "やくす", "dịch .")]},
    "君": {"viet": "QUÂN", "meaning_vi": "Vua, người làm chủ cả một nước.", "meo": "", "reading": "きみ -ぎみ", "vocab": [("君", "きみ", "em"), ("君", "くん", "cậu; bạn; mày")]},
    "笑": {"viet": "TIẾU", "meaning_vi": "Cười, vui cười.", "meo": "", "reading": "わら.う え.む", "vocab": [("笑い", "わらい", "tiếng cười; sự chê cười"), ("笑う", "わらう", "cười; mỉm cười")]},
    "危": {"viet": "NGUY", "meaning_vi": "Cao, ở nơi cao mà ghê sợ gọi là nguy.", "meo": "", "reading": "あぶ.ない あや.うい あや.ぶむ", "vocab": [("危地", "きち", "sự nguy hiểm"), ("安危", "あんき", "thiên mệnh")]},
    "犯": {"viet": "PHẠM", "meaning_vi": "Xâm phạm, cái cứ không nên xâm vào mà cứ xâm vào gọi là phạm. Như can phạm [干犯], mạo phạm [冒犯], v.v.", "meo": "", "reading": "おか.す", "vocab": [("犯す", "おかす", "vi phạm; xâm phạm"), ("主犯", "しゅはん", "cái đầu (người")]},
    "器": {"viet": "KHÍ", "meaning_vi": "Đồ. Như khí dụng [器用] đồ dùng.", "meo": "", "reading": "うつわ", "vocab": [("器", "うつわ", "chậu; bát..."), ("不器", "ふき", "Sự vụng về .")]},
    "引": {"viet": "DẪN, DẤN", "meaning_vi": "Dương cung. Như dẫn mãn [引滿] dương hết cữ cung.", "meo": "", "reading": "ひ.く ひ.き ひ.き- -び.き ひ.ける", "vocab": [("引き", "ひき", "giật ."), ("引く", "ひく", "chăng")]},
    "第": {"viet": "ĐỆ", "meaning_vi": "Thứ đệ. Như đệ nhất [第一] thứ nhất, đệ nhị [第二] thứ hai, v.v.", "meo": "", "reading": "ダイ テイ", "vocab": [("第", "だい", "thứ"), ("第一", "だいいち", "đầu tiên; quan trọng")]},
    "費": {"viet": "PHÍ, BỈ", "meaning_vi": "Tiêu phí.", "meo": "", "reading": "つい.やす つい.える", "vocab": [("費", "ひ", "phí"), ("費え", "ついえ", "chi phí lãng phí .")]},
    "法": {"viet": "PHÁP", "meaning_vi": "Phép, có khuôn phép nhất định để cho người tuân theo được gọi là pháp. Như pháp điển [法典] bộ luật pháp, pháp quy [法規] khuôn phép, pháp luật [法律] phép luật, v.v.", "meo": "", "reading": "のり", "vocab": [("法", "ほう", "lễ pháp"), ("不法", "ふほう", "không có pháp luật; hỗn độn; vô trật tự")]},
    "飛": {"viet": "PHI", "meaning_vi": "Bay. Loài chim và loài sậu cất cánh bay cao gọi là phi.", "meo": "", "reading": "と.ぶ と.ばす -と.ばす", "vocab": [("飛ぶ", "とぶ", "bay nhảy"), ("勇飛", "いさむひ", "cú nhảy xa")]},
    "戸": {"viet": "HỘ", "meaning_vi": "Hộ khẩu.", "meo": "", "reading": "と", "vocab": [("戸", "と", "cánh cửa"), ("一戸", "いちこ", "hộ")]},
    "能": {"viet": "NĂNG, NAI, NẠI", "meaning_vi": "Tài năng. Như năng viên [能員] chức quan có tài.", "meo": "", "reading": "よ.く", "vocab": [("能", "のう", "hiệu lực; hiệu quả"), ("能く", "よく", "giỏi; đẹp; hay")]},
    "久": {"viet": "CỬU", "meaning_vi": "Lâu, nói thì giờ đã lâu. Như cửu mộ [久慕] mến đã lâu , cửu ngưỡng [久仰] kính đã lâu.", "meo": "", "reading": "ひさ.しい", "vocab": [("恒久", "こうきゅう", "sự vĩnh cửu; cái không thay đổi; sự vĩnh viễn"), ("久しい", "ひさしい", "đã lâu; đã bao lâu nay")]},
    # "事" → already in main DB
    "関": {"viet": "QUAN", "meaning_vi": "Hải quan,", "meo": "", "reading": "せき -ぜき かか.わる からくり かんぬき", "vocab": [("関", "せき", "cổng; ba-ri-e"), ("関与", "かんよ", "sự tham dự; tham dự; sự tham gia; sự liên quan; liên quan")]},
    "舟": {"viet": "CHU", "meaning_vi": "Thuyền. Các cái như thuyền, bè dùng qua sông qua nước đều gọi là chu. Nguyễn Du [阮攸] : Thiên địa thiên chu phù tự diệp, Văn chương tàn tức nhược như ti [天地扁舟浮以葉, 文章殘息弱如絲] (Chu hành tức sự [舟行即事]) Chiếc thuyền con như chiếc lá nổi giữa đất trời, Hơi tàn văn chương yếu ớt như tơ. Bùi Giáng dịch thơ :  Thuyền con chiếc lá giữa trời, thơ văn tiếng thở như lời tơ than.", "meo": "", "reading": "ふね ふな- -ぶね", "vocab": [("舟", "ふね", "tàu; thuyền ."), ("舟人", "ふなびと", "lính thuỷ")]},
    "程": {"viet": "TRÌNH", "meaning_vi": "Khuôn phép. Như chương trình [章程], trình thức [程式] đều nghĩa là cái khuôn phép để làm việc cả.", "meo": "", "reading": "ほど -ほど", "vocab": [("程", "ほど", "bằng"), ("程々", "ほどほど", "ở một mức độ vừa phải; không nhiều lắm; trầm lặng")]},
    "直": {"viet": "TRỰC", "meaning_vi": "Thẳng. Như trực tuyến [直線] đường thẳng.", "meo": "", "reading": "ただ.ちに なお.す -なお.す なお.る なお.き す.ぐ", "vocab": [("直", "じき", "gần; sớm"), ("直", "ちょく", "trực tiếp; ngay")]},
    "置": {"viet": "TRÍ", "meaning_vi": "Để, cầm đồ gì để yên vào đâu đều gọi là trí.", "meo": "", "reading": "お.く -お.き", "vocab": [("置く", "おく", "bố trí (người)"), ("並置", "へいち", "sự đặt cạnh nhau")]},
    "植": {"viet": "THỰC, TRĨ", "meaning_vi": "Các loài có rễ. Như thực vật [植物] các cây cỏ.", "meo": "", "reading": "う.える う.わる", "vocab": [("入植", "にゅうしょく", "sự nhập cư ."), ("植字", "しょくじ", "xếp chữ")]},
    "録": {"viet": "LỤC", "meaning_vi": "Dị dạng của chữ 录", "meo": "", "reading": "しる.す と.る", "vocab": [("付録", "ふろく", "phụ lục ."), ("余録", "よろく", "tiếng đồn")]},
    "押": {"viet": "ÁP", "meaning_vi": "Ký. Như hoa áp [花押] ký chữ để làm ghi.", "meo": "", "reading": "お.す お.し- お.っ- お.さえる おさ.える", "vocab": [("押え", "おさえ", "quyền hành"), ("押し", "おし", "sự xô")]},
    # "無" → already in main DB
    "座": {"viet": "TỌA", "meaning_vi": "Ngôi, tòa, nhà lớn, chỗ để ngồi gọi là tọa. Trần Nhân Tông [陳仁宗] : Huân tận thiên đầu mãn tọa hương [薰尽千頭满座香] (Đề Phổ Minh tự thủy tạ [題普明寺水榭]) Đốt hết nghìn nén hương mùi thơm bay đầy nhà.", "meo": "", "reading": "すわ.る", "vocab": [("座", "ざ", "chỗ ngồi; địa vị; không khí; cung (trong bói toán)"), ("座り", "すわり", "sự ngồi; sự đặt ngồi")]},
    "卒": {"viet": "TỐT, TUẤT, THỐT", "meaning_vi": "Quân lính. Như binh tốt [兵卒] binh lính, tẩu tốt [走卒] lính hầu.", "meo": "", "reading": "そっ.する お.える お.わる ついに にわか", "vocab": [("万卒", "まんそつ", "lính gác"), ("卒中", "そっちゅう", "chứng ngập máu .")]},
    "差": {"viet": "SOA, SI, SAI, SÁI", "meaning_vi": "Sai nhầm.", "meo": "", "reading": "さ.す さ.し", "vocab": [("差", "さ", "sự khác biệt; khoảng cách"), ("差す", "さす", "giương (ô); giơ (tay) .")]},
    "綿": {"viet": "MIÊN", "meaning_vi": "Bông mới. Cho kén vào nước sôi rồi gỡ ra chỗ nào săn đẹp gọi là miên [綿], chỗ nào sù sì gọi là nhứ [絮].", "meo": "", "reading": "わた", "vocab": [("綿", "めん", "bông; tơ sống"), ("綿", "わた", "bông gòn")]},
    "形": {"viet": "HÌNH", "meaning_vi": "Hình thể.", "meo": "", "reading": "かた -がた かたち なり", "vocab": [("形", "かたち", "hình dáng; kiểu"), ("丸形", "まるがた", "đường tròn")]},
    "型": {"viet": "HÌNH", "meaning_vi": "Cái khuôn. Cái khuôn bằng đất để đúc các đồ gọi là hình.", "meo": "", "reading": "かた -がた", "vocab": [("型", "かた", "cách thức"), ("丸型", "まるがた", "đường tròn")]},
    "杯": {"viet": "BÔI", "meaning_vi": "Cái chén.", "meo": "", "reading": "さかずき", "vocab": [("杯", "さかずき", "cốc; chén"), ("w杯", "ダブルはい", "cúp bóng đá thế giới .")]},
    "否": {"viet": "PHỦ, BĨ, PHẦU", "meaning_vi": "Không. Như thường kì chỉ phủ [嘗其旨否] nếm xem ngon không ?", "meo": "", "reading": "いな いや", "vocab": [("否", "いいえ", "không"), ("否", "いえ", "không")]},
    "満": {"viet": "MÃN", "meaning_vi": "Thỏa mãn, mãn nguyện.", "meo": "", "reading": "み.ちる み.つ み.たす", "vocab": [("不満", "ふまん", "bất bình; bất mãn"), ("満了", "まんりょう", "sự mãn hạn; sự chấm dứt; sự kết thúc .")]},
    "留": {"viet": "LƯU", "meaning_vi": "Lưu lại, muốn đi mà tạm ở lại gọi là lưu.", "meo": "", "reading": "と.める と.まる とど.める とど.まる るうぶる", "vocab": [("留め", "とめ", "lý lẽ vững chắc"), ("係留", "けいりゅう", "sự thả neo")]},
    "貿": {"viet": "MẬU", "meaning_vi": "Đổi lẫn cho nhau. Như mậu dịch [貿易] mua bán.", "meo": "", "reading": "ボウ", "vocab": [("貿易", "ぼうえき", "giao dịch"), ("貿易会", "ぼうえきかい", "hội mậu dịch .")]},
    "逆": {"viet": "NGHỊCH, NGHỊNH", "meaning_vi": "Trái. Trái lại với chữ thuận [順]. Phàm cái gì không thuận đều gọi là nghịch cả.", "meo": "", "reading": "さか さか.さ さか.らう", "vocab": [("逆", "ぎゃく", "kình địch"), ("逆さ", "さかさ", "ngược; sự ngược lại; sự đảo lộn")]},
    "官": {"viet": "QUAN", "meaning_vi": "Quan lại, cơ quan", "meo": "Mái nhà (宀) che chở cho mọi người (人) trong cơ quan (官).", "reading": "かん", "vocab": [("官庁", "かんちょう", "Cơ quan chính phủ"), ("官僚", "かんりょう", "Quan liêu")]},
    "管": {"viet": "QUẢN", "meaning_vi": "Cái sáo nhỏ. Nguyễn Du [阮攸] : Quản huyền nhất biến tạp tân thanh [管絃一變雜新聲] (Thăng Long [昇龍]) Đàn sáo một loạt thay đổi, chen vào những thanh điệu mới.", "meo": "", "reading": "くだ", "vocab": [("管", "かん", "ống"), ("管", "くだ", "kèn")]},
    "追": {"viet": "TRUY, ĐÔI", "meaning_vi": "Đuổi theo. Như truy tung [追蹤] theo hút, theo vết chân mà đuổi.", "meo": "", "reading": "お.う", "vocab": [("追う", "おう", "bận rộn; nợ ngập đầu ngập cổ"), ("追伸", "ついしん", "tái bút")]},
    "師": {"viet": "SƯ", "meaning_vi": "Nhiều, đông đúc. Như chỗ đô hội trong nước gọi là kinh sư [京師] nghĩa là chỗ to rộng và đông người.", "meo": "", "reading": "いくさ", "vocab": [("師", "し", "giáo viên"), ("京師", "けいし", "thủ đô")]},
    "指": {"viet": "CHỈ", "meaning_vi": "Ngón tay. Tay có năm ngón, ngón tay cái gọi là cự chỉ [巨指] hay mẫu chỉ [拇指], ngón tay trỏ gọi là thực chỉ [食指], ngón tay giữa gọi là tướng chỉ [將指], ngón tay đeo nhẫn gọi là vô danh chỉ [無名指], ngón tay út gọi là tiểu chỉ [小指].", "meo": "", "reading": "ゆび さ.す -さ.し", "vocab": [("指", "ゆび", "ngón"), ("指す", "さす", "chỉ ra; chỉ ra")]},
    "感": {"viet": "CẢM", "meaning_vi": "Cảm hóa, lấy lời nói sự làm của mình làm cảm động được người gọi là cảm hóa [感化] hay cảm cách [感格].", "meo": "", "reading": "カン", "vocab": [("感", "かん", "sự sờ mó"), ("感じ", "かんじ", "tri giác; cảm giác")]},
    "認": {"viet": "NHẬN", "meaning_vi": "Biện rõ, nhận biết. Như nhận minh [認明] nhận rõ ràng.", "meo": "", "reading": "みと.める したた.める", "vocab": [("認め", "みとめ", "sự thừa nhận; sự coi trọng ."), ("信認", "しんにん", "nhận")]},
    "旧": {"viet": "CỰU", "meaning_vi": "Giản thể của chữ [舊].", "meo": "", "reading": "ふる.い もと", "vocab": [("旧", "きゅう", "âm lịch"), ("旧例", "きゅうれい", "sự truyền miệng (truyện cổ tích")]},
    "児": {"viet": "NHI", "meaning_vi": "Nhi đồng, hài nhi", "meo": "", "reading": "こ -こ -っこ", "vocab": [("児", "じ", "trẻ nhỏ"), ("乳児", "にゅうじ", "con mọn")]},
    # "度" → already in main DB
    "席": {"viet": "TỊCH", "meaning_vi": "Cái chiếu. Như thảo tịch [草席] chiếu cỏ.", "meo": "", "reading": "むしろ", "vocab": [("席", "せき", "chỗ ngồi ."), ("一席", "いっせき", "sự ngồi; sự đặt ngồi")]},
    "成": {"viet": "THÀNH", "meaning_vi": "Nên, thành tựu, phàm làm công việc gì đến lúc xong đều gọi là thành. Như làm nhà xong gọi là lạc thành [落成], làm quan về hưu gọi là hoạn thành [宦成], v.v.", "meo": "", "reading": "な.る な.す -な.す", "vocab": [("成す", "なす", "hoàn thành; làm xong"), ("成る", "なる", "thành")]},
    "械": {"viet": "GIỚI", "meaning_vi": "Đồ khí giới. Như binh giới [兵械] đồ binh.", "meo": "", "reading": "かせ", "vocab": [("器械", "きかい", "khí giới; dụng cụ; công cụ"), ("機械", "きかい", "bộ máy")]},
    "式": {"viet": "THỨC", "meaning_vi": "Phép. Sự gì đáng làm khuôn phép gọi là túc thức [足式].", "meo": "", "reading": "シキ", "vocab": [("式", "しき", "hình thức; kiểu; lễ; nghi thức"), ("一式", "いっしき", "tất cả")]},
    "収": {"viet": "THU", "meaning_vi": "Thu nhập, thu nhận,", "meo": "", "reading": "おさ.める おさ.まる", "vocab": [("収入", "しゅうにゅう", "thu nhập ."), ("収受", "しゅうじゅ", "sự nhận")]},
    "担": {"viet": "ĐAM, ĐẢM", "meaning_vi": "Giản thể của chữ [擔].", "meo": "", "reading": "かつ.ぐ にな.う", "vocab": [("担", "たん", "bờ giậu"), ("担う", "になう", "cáng đáng")]},
    "洗": {"viet": "TẨY, TIỂN", "meaning_vi": "Giặt, rửa.", "meo": "", "reading": "あら.う", "vocab": [("洗う", "あらう", "giặt; rửa; tắm gội"), ("洗剤", "せんざい", "bột làm bánh")]},
    "熱": {"viet": "NHIỆT", "meaning_vi": "Nóng. Như nhiệt thiên [熱天] trời mùa nóng, ngày hè.", "meo": "", "reading": "あつ.い", "vocab": [("熱", "ねつ", "nhiệt độ"), ("熱い", "あつい", "nóng; nóng bỏng; oi bức; thân thiện; nhiệt tình")]},
    "陸": {"viet": "LỤC", "meaning_vi": "Đồng bằng cao ráo, đất liền. Vì nói phân biệt với bể nên năm châu gọi là đại lục [大陸] cõi đất liền lớn.", "meo": "", "reading": "おか", "vocab": [("陸", "りく", "lục địa; mặt đất; đất liền"), ("陸上", "りくじょう", "trên mặt đất; trên đất liền")]},
    "巨": {"viet": "CỰ", "meaning_vi": "to lớn, khổng lồ", "meo": "Hình tượng người thợ xây (工) đội mũ (冖) ĐANG TỪ TỪ PHÌNH TO RA.", "reading": "きょ", "vocab": [("巨大", "きょだい", "khổng lồ"), ("巨人", "きょじん", "người khổng lồ")]},
    "臣": {"viet": "THẦN", "meaning_vi": "Bầy tôi, quan ở trong nước có vua gọi là thần. Ngày xưa gọi những kẻ làm quan hai họ là nhị thần [貳臣].", "meo": "", "reading": "シン ジン", "vocab": [("下臣", "しもしん", "chưa hầu; phong hầu"), ("臣下", "しんか", "lão bộc; quản gia; người hầu cận; người tùy tùng .")]},
    "設": {"viet": "THIẾT", "meaning_vi": "Sắp bày, đặt bày. Như trần thiết [陳設] bày đặt. Nhà vẽ tô mùi (màu) thuốc gọi là thiết sắc [設色].", "meo": "", "reading": "もう.ける", "vocab": [("設け", "もうけ", "sự soạn"), ("付設", "ふせつ", "phụ vào")]},
    "投": {"viet": "ĐẦU", "meaning_vi": "Ném. Như đầu hồ [投壺] ném thẻ vào trong hồ.", "meo": "", "reading": "な.げる -な.げ", "vocab": [("投げ", "なげ", "Cú ném; cú quật"), ("投下", "とうか", "việc thả (quân lính")]},
    "役": {"viet": "DỊCH", "meaning_vi": "Đi thú ngoài biên thùy. Đi thú xa gọi là viễn dịch [遠役].", "meo": "", "reading": "ヤク エキ", "vocab": [("役", "えき", "chiến tranh; cuộc chiến; chiến dịch"), ("役", "やく", "giá trị hoặc lợi ích; tính hữu ích")]},
    "段": {"viet": "ĐOẠN", "meaning_vi": "Chia từng đoạn, vải lụa định mấy thước cắt làm một, mỗi tấm gọi là đoạn.", "meo": "", "reading": "ダン タン", "vocab": [("段", "だん", "bước"), ("段々", "だんだん", "dần dần")]},
    "缶": {"viet": "PHỮU, PHẪU, PHŨ", "meaning_vi": "Đồ sành. Như cái vò cái chum, v.v.", "meo": "", "reading": "かま", "vocab": [("缶", "かん", "bi đông; ca; lon; cặp lồng"), ("汽缶", "きかん", "người đun")]},
    "候": {"viet": "HẬU", "meaning_vi": "Dò ngóng. Như vấn hậu [問候] tìm hỏi thăm bạn, trinh hậu [偵候] dò xét, đều là cái ý nghĩa lặng đợi dò xét cả.", "meo": "", "reading": "そうろう", "vocab": [("候", "こう", "thời tiết; khí hậu; mùa"), ("兆候", "ちょうこう", "triệu chứng; dấu hiệu")]},
    "角": {"viet": "GIÁC, GIỐC", "meaning_vi": "Cái sừng, cái sừng của các giống thú. Như tê giác [犀角] sừng con tên ngưu.", "meo": "", "reading": "かど つの", "vocab": [("角", "かく", "góc"), ("角", "つの", "sừng .")]},
    "解": {"viet": "GIẢI, GIỚI, GIÁI", "meaning_vi": "Bửa ra, mổ ra. Dùng cưa xẻ gỗ ra gọi là giải mộc [解木]. Mổ xẻ người để chữa bệnh gọi là giải phẩu [解剖].", "meo": "", "reading": "と.く と.かす と.ける ほど.く ほど.ける わか.る さと.る", "vocab": [("解く", "ほどく", "mở ra; cởi bỏ ."), ("解く", "とく", "giải đáp; cởi bỏ")]},
    "島": {"viet": "ĐẢO", "meaning_vi": "Bãi bể, cái cù lao, trong bể có chỗ đất cạn gọi là đảo.", "meo": "", "reading": "しま", "vocab": [("島", "しま", "đảo"), ("島々", "しまじま", "những hòn đảo")]},
}




# ── N2 Kanji (505 chữ) ──────────────────────────────────────────────────────
N2_VI: dict[str, dict] = {
    "穴": {"viet": "HUYỆT", "meaning_vi": "Hang, lỗ", "meo": "Nhớ theo hình dạng: Mái nhà (宀) bị sập, tám (八) người phải chui vào hang (穴) trú ẩn.", "reading": "あな", "vocab": [("穴", "あな", "Hang, lỗ"), ("風穴", "ふうけつ", "Hang gió")]},
    "込": {"viet": "", "meaning_vi": "chen chúc, đông đúc, đầy ứ, nhồi nhét, lấp kín, (cho, tính, lắp...) vào", "meo": "", "reading": "-こ.む こ.む こ.み -こ.み こ.める", "vocab": [("込む", "こむ", "đông đúc"), ("仕込", "しこみ", "sự dạy dỗ")]},
    "片": {"viet": "PHIẾN", "meaning_vi": "Mảnh, vật gì mỏng mà phẳng đều gọi là phiến. Như mộc phiến [木片] tấm ván, chỉ phiến [紙片] mảnh giấy.", "meo": "", "reading": "かた- かた", "vocab": [("一片", "いっぺん", "miếng mỏng"), ("片側", "かたがわ", "một bên")]},
    "散": {"viet": "TÁN, TẢN", "meaning_vi": "Tan. Như vân tán [雲散] mây tan.", "meo": "", "reading": "ち.る ち.らす -ち.らす ち.らかす ち.らかる ち.らばる ばら ばら.ける", "vocab": [("散々", "さんざん", "gay go; khốc liệt; dữ dội; buồn thảm"), ("散る", "ちる", "héo tàn")]},
    "背": {"viet": "BỐI, BỘI", "meaning_vi": "Vai, hai bên sau lưng ngang với ngực gọi là bối. Như bối tích [背脊] xương sống lưng, chuyển bối [轉背] xoay lưng, ý nói rất mau chóng, khoảnh khắc.", "meo": "", "reading": "せ せい そむ.く そむ.ける", "vocab": [("背", "せい", "lưng ."), ("背く", "そむく", "bội phản")]},
    "胃": {"viet": "VỊ", "meaning_vi": "Dạ dày, dùng để đựng và tiêu hóa đồ ăn. Như vị dịch [胃液] chất lỏng do dạ dày tiết ra để tiêu hóa đồ ăn.", "meo": "", "reading": "イ", "vocab": [("胃", "い", "dạ dầy"), ("胃弱", "いじゃく", "bội thực")]},
    "貨": {"viet": "HÓA", "meaning_vi": "Của. Như hóa tệ [貨幣] của cải. Phàm vật gì có thể đổi lấy tiền được đều gọi là hóa.", "meo": "", "reading": "たから", "vocab": [("外貨", "がいか", "khoa ngoại"), ("貨幣", "かへい", "tiền tệ; tiền; đồng tiền")]},
    "靴": {"viet": "NGOA", "meaning_vi": "Bì ngoa [皮靴] giày ủng làm bằng da. Cũng như chữ ngoa [鞾].", "meo": "", "reading": "くつ", "vocab": [("靴", "くつ", "giày; dép; guốc"), ("靴下", "くつした", "bít tất")]},
    "含": {"viet": "HÀM", "meaning_vi": "Ngậm, ngậm ở trong mồm không nhả không nuốt là hàm.", "meo": "", "reading": "ふく.む ふく.める", "vocab": [("含み", "ふくみ", "sự lôi kéo vào; sự liên can"), ("含む", "ふくむ", "bao gồm")]},
    "令": {"viet": "LỆNH, LINH", "meaning_vi": "Mệnh lệnh, những điều mà chính phủ đem ban bố cho dân biết gọi là lệnh.", "meo": "", "reading": "レイ", "vocab": [("令", "れい", "lệnh; mệnh lệnh; chỉ thị ."), ("仮令", "たとえ", "ví dụ; nếu như; dù cho; ngay cả nếu; tỉ như")]},
    "領": {"viet": "LĨNH", "meaning_vi": "Cái cổ. Như Mạnh Tử [孟子] nói Tắc thiên hạ chi dân, giai dẫn lĩnh nhi vọng chi hĩ [則天下之民皆引領而望之矣] Thì dân trong thiên hạ đều nghển cổ mà trông mong vậy.", "meo": "", "reading": "えり", "vocab": [("主領", "しゅりょう", "cái đầu (người"), ("領事", "りょうじ", "lãnh sự .")]},
    "齢": {"viet": "LINH", "meaning_vi": "Tuổi", "meo": "", "reading": "よわい とし", "vocab": [("余齢", "よよわい", "tuổi thọ trung bình"), ("壮齢", "そうれい", "sự chôn cất")]},
    "迷": {"viet": "MÊ", "meaning_vi": "Lạc. Như mê lộ [迷路] lạc đường.", "meo": "", "reading": "まよ.う", "vocab": [("迷い", "まよい", "hesitance"), ("迷う", "まよう", "bị lúng túng; không hiểu")]},
    "菊": {"viet": "CÚC", "meaning_vi": "Hoa cúc. Đào Uyên Minh [陶淵明] : Tam kính tựu hoang, tùng cúc do tồn [三 徑 就 荒，松 菊 猶 存] (Quy khứ lai từ [歸去來辭]) Ra lối nhỏ đến vườn hoang, hàng tùng hàng cúc hãy còn đây.", "meo": "", "reading": "キク", "vocab": [("菊", "きく", "cúc"), ("春菊", "しゅんぎく", "cải cúc")]},
    "申": {"viet": "THÂN", "meaning_vi": "Chi Thân, một chi trong mười hai chi. Từ 3 giờ chiều đến năm giờ chiều gọi là giờ Thân.", "meo": "", "reading": "もう.す もう.し- さる", "vocab": [("申す", "もうす", "nói là; được gọi là; tên là"), ("上申", "じょうしん", "sự đề xuất; sự báo cáo với cấp trên. .")]},
    "栄": {"viet": "VINH", "meaning_vi": "Vinh quang", "meo": "", "reading": "さか.える は.え -ば.え は.える え", "vocab": [("栄え", "はえ", "sự phồn vinh"), ("栄え", "さかえ", "sự phồn vinh")]},
    "杉": {"viet": "SAM", "meaning_vi": "Cây sam, là một thứ gỗ thông dùng rất nhiều việc.", "meo": "", "reading": "すぎ", "vocab": [("杉", "すぎ", "cây tuyết tùng ở Nhật ."), ("糸杉", "いとすぎ", "cây bách")]},
    "条": {"viet": "ĐIỀU, THIÊU, ĐIÊU", "meaning_vi": "Tục dùng như chữ điều [條].", "meo": "", "reading": "えだ すじ", "vocab": [("一条", "いちじょう", "đường sọc"), ("条令", "じょうれい", "sự điều chỉnh")]},
    "憶": {"viet": "ỨC", "meaning_vi": "ức, nhớ", "meo": "Trong TÂM (心) có Ý (意) muốn ghi ức.", "reading": "おく", "vocab": [("記憶", "きおく", "ký ức, trí nhớ"), ("追憶", "ついおく", "sự hồi tưởng, nhớ lại")]},
    "章": {"viet": "CHƯƠNG", "meaning_vi": "Văn chương, chương mạch. Như văn chương [文章] lời nói hay đẹp được viết thành chữ, thành bài.", "meo": "", "reading": "ショウ", "vocab": [("章", "しょう", "chương; hồi (sách)"), ("勲章", "くんしょう", "huân chương")]},
    "灰": {"viet": "HÔI, KHÔI", "meaning_vi": "Tro. Lý Thương Ẩn [李商隱] : Lạp cự thành hôi lệ thủy can [蠟炬成灰淚始乾] (Vô đề [無題]) Ngọn nến thành tro mới khô nước mắt.", "meo": "", "reading": "はい", "vocab": [("灰", "かい", "tàn"), ("灰", "はい", "tro tàn .")]},
    "炭": {"viet": "THÁN", "meaning_vi": "Than.", "meo": "", "reading": "すみ", "vocab": [("亜炭", "あたん", "than bùn"), ("炭俵", "すみだわら", "bao tải than .")]},
    "粒": {"viet": "LẠP", "meaning_vi": "Hạt gạo, hạt lúa, vật gì nhỏ mà rời từng hạt đều gọi là lạp [粒].", "meo": "", "reading": "つぶ", "vocab": [("粒", "つぶ", "hạt; hột"), ("一粒", "ひとつぶ", "Một hạt")]},
    "舌": {"viet": "THIỆT", "meaning_vi": "Lưỡi.", "meo": "", "reading": "した", "vocab": [("舌", "した", "lưỡi ."), ("両舌", "りょうぜつ", "trò hai mang")]},
    "乱": {"viet": "LOẠN", "meaning_vi": "Dùng như chữ loạn [亂].", "meo": "", "reading": "みだ.れる みだ.る みだ.す みだ おさ.める わた.る", "vocab": [("乱す", "みだす", "chen ngang"), ("乱れ", "みだれ", "sự mất trật tự")]},
    "酢": {"viet": "TẠC", "meaning_vi": "Khách rót rượu cho chủ, phàm đã nhận cái gì của người mà lại lấy vật khác trả lại đều gọi là tạc. Hai bên cùng đưa lẫn cho nhau gọi là thù tạc [酬酢].", "meo": "", "reading": "す", "vocab": [("酢", "す", "giấm ."), ("甘酢", "あまず", "dấm ngọt .")]},
    "簡": {"viet": "GIẢN", "meaning_vi": "Cái thẻ tre. Đời xưa chưa có giấy viết vào thẻ tre gọi là gian trát [簡札], vì thế nên gọi sách vở là giản. Như đoạn giản tàn biên [斷簡殘編] sách vở đứt nát. Bây giờ gọi phong thơ là thủ giản [手簡] là vì lẽ đó.", "meo": "", "reading": "えら.ぶ ふだ", "vocab": [("了簡", "りょうけん", "quan niệm"), ("簡便", "かんべん", "thuận tiện")]},
    "宝": {"viet": "BẢO", "meaning_vi": "Như chữ bảo [寶].", "meo": "", "reading": "たから", "vocab": [("宝", "たから", "bảo ."), ("七宝", "しっぽう", "thất bảo")]},
    "溶": {"viet": "DONG, DUNG", "meaning_vi": "Dong dong [溶溶] nước mông mênh.", "meo": "", "reading": "と.ける と.かす と.く", "vocab": [("溶く", "とく", "làm tan ra ."), ("溶剤", "ようざい", "có khả năng hoà tan")]},
    "導": {"viet": "ĐẠO", "meaning_vi": "Dẫn đưa. Đi trước đường gọi là tiền đạo [前導].", "meo": "", "reading": "みちび.く", "vocab": [("導き", "みちびき", "sự chỉ đạo"), ("導く", "みちびく", "đạo")]},
    "損": {"viet": "TỔN", "meaning_vi": "Bớt. Như tổn thượng ích hạ [損上益下] bớt kẻ trên thêm kẻ dưới.", "meo": "", "reading": "そこ.なう そこな.う -そこ.なう そこ.ねる -そこ.ねる", "vocab": [("損", "そん", "lỗ ."), ("損う", "そこなう", "làm hại; làm tổn hại; làm đau; làm bị thương")]},
    "姓": {"viet": "TÍNH", "meaning_vi": "Họ. Như tính danh [姓名] họ và tên.", "meo": "", "reading": "セイ ショウ", "vocab": [("姓", "せい", "họ"), ("同姓", "どうせい", "sự cùng họ .")]},
    "詩": {"viet": "THI", "meaning_vi": "Thơ, văn có vần gọi là thơ. Ngày xưa hay đặt mỗi câu bốn chữ, về sau hay dùng lối đặt năm chữ hay bảy chữ gọi là thơ ngũ ngôn [五言], thơ thất ngôn [七言].", "meo": "", "reading": "うた", "vocab": [("詩", "し", "thơ"), ("詩人", "しじん", "nhà thơ")]},
    "改": {"viet": "CẢI", "meaning_vi": "Đổi. Như cải tạo [改造] làm lại, cải quá [改過] đổi lỗi, v.v.", "meo": "", "reading": "あらた.める あらた.まる", "vocab": [("改修", "かいしゅう", "sự sửa chữa; sự cải tiến; sửa chữa; cải tiến; nâng cấp; sự cải tạo; cải tạo"), ("改号", "かいごう", "cuộc mít tinh")]},
    # "配" → already in main DB
    "幅": {"viet": "PHÚC, BỨC", "meaning_vi": "Bức, một tiếng dùng để đo vải lụa. Như kỉ phúc [幾幅] mấy bức ?", "meo": "", "reading": "はば", "vocab": [("幅", "はば", "chiều rộng; chiều ngang"), ("一幅", "いちはば", "cuộn giấy")]},
    "枝": {"viet": "CHI", "meaning_vi": "cành cây", "meo": "Cây mọc ra MƯỜI nhánh.", "reading": "えだ", "vocab": [("枝", "えだ", "cành cây"), ("枝豆", "えだまめ", "đậu nành Nhật Bản")]},
    "漁": {"viet": "NGƯ", "meaning_vi": "Bắt cá, đánh cá. Âu Dương Tu [歐陽修] : Lâm khê nhi ngư [臨溪而漁] (Túy Ông đình ký [醉翁亭記]) Vào ngòi câu cá.", "meo": "", "reading": "あさ.る", "vocab": [("漁", "りょう", "sự đánh cá"), ("漁り", "いさり", "sự nhìn để tìm")]},
    "雷": {"viet": "LÔI", "meaning_vi": "Sấm. Như lôi điện [雷電] sấm sét.", "meo": "", "reading": "かみなり いかずち いかづち", "vocab": [("雷", "かみなり", "sấm sét"), ("雷光", "らいこう", "chớp")]},
    "衆": {"viet": "CHÚNG", "meaning_vi": "Dị dạng của chữ [众]", "meo": "", "reading": "おお.い", "vocab": [("衆", "しゅう", "công chúng ."), ("会衆", "かいしゅう", "thính giả; công chúng; mọi người")]},
    "益": {"viet": "ÍCH", "meaning_vi": "Thêm lên, phàm cái gì có tiến bộ hơn đều gọi là ích.", "meo": "", "reading": "ま.す", "vocab": [("益", "えき", "lợi ích; tác dụng"), ("益々", "ますます", "ngày càng")]},
    "沖": {"viet": "TRÙNG, XUNG", "meaning_vi": "Rỗng không, trong lòng lặng lẽ rỗng không, không cạnh tranh gì gọi là trùng. Như khiêm trùng [謙沖] nhún nhường, lặng lẽ.", "meo": "", "reading": "おき おきつ ちゅう.する わく", "vocab": [("沖", "おき", "biển khơi; khơi"), ("沖合", "おきあい", "ngoài khơi")]},
    "患": {"viet": "HOẠN", "meaning_vi": "Lo. Như hoạn đắc hoạn thất [患得患失] lo được lo mất.", "meo": "", "reading": "わずら.う", "vocab": [("患い", "わずらい", "bệnh ."), ("患う", "わずらう", "bị ốm; bị bệnh; ngã bệnh .")]},
    "疲": {"viet": "BÌ", "meaning_vi": "Mỏi mệt. Như cân bì lực tận [筋疲力盡] gân cốt mệt nhoài.", "meo": "", "reading": "つか.れる -づか.れ つか.らす", "vocab": [("疲れ", "つかれ", "sự mệt mỏi"), ("疲労", "ひろう", "mệt mỏi")]},
    "被": {"viet": "BỊ, BÍ", "meaning_vi": "Áo ngủ.", "meo": "", "reading": "こうむ.る おお.う かぶ.る かぶ.せる", "vocab": [("被い", "おおい", "áo khoác"), ("被う", "おおう", "bao bọc")]},
    "波": {"viet": "BA", "meaning_vi": "Sóng nhỏ. Sóng nhỏ gọi là ba [波], sóng lớn gọi là lan [瀾]. Văn bài gì có từng thứ nẩy ra gọi là ba lan [波瀾].", "meo": "", "reading": "なみ", "vocab": [("波", "なみ", "làn sóng"), ("中波", "ちゅうは", "sóng tầm trung; sóng vừa .")]},
    "破": {"viet": "PHÁ", "meaning_vi": "Phá vỡ. Như phá hoại [破壞], phá toái [破碎], phá trận [破陣], phá thành [破城], v.v.", "meo": "", "reading": "やぶ.る やぶ.れる わ.れる", "vocab": [("破く", "やぶく", "nước mắt"), ("破る", "やぶる", "bị rách")]},
    "士": {"viet": "SĨ", "meaning_vi": "Học trò, những người nghiên cứu học vấn đều gọi là sĩ.", "meo": "", "reading": "さむらい", "vocab": [("一士", "いちし", "ngón tay"), ("人士", "じんし", "nhân sĩ .")]},
    "誌": {"viet": "CHÍ", "meaning_vi": "Ghi nhớ. Như chí chi bất vong [誌之不忘] ghi nhớ chẳng quên.", "meo": "", "reading": "シ", "vocab": [("誌", "し", "báo"), ("誌上", "しじょう", "trên tạp chí")]},
    "梅": {"viet": "MAI", "meaning_vi": "Cây mơ, đầu xuân đã nở hoa, có hai thứ trắng và đỏ. Thứ trắng gọi là lục ngạc mai [綠萼梅], nở hết hoa rồi mới nẩy lá, quả chua, chín thì sắc vàng. Kinh Thư có câu nhược tác hòa canh, nhĩ duy diêm mai [若作和羹, 爾惟鹽梅] bằng nấu canh ăn, bui dùng muối mơ. Nay gọi quan Tể tướng là điều mai [調梅] hay hòa mai [和梅] là bởi ý đó. Kinh Thi có thơ phiếu mai [摽梅] mai rụng, nói sự trai gái lấy nhau cập thời, nay gọi con gái sắp đi lấy chồng là bởi cớ đó.", "meo": "", "reading": "うめ", "vocab": [("梅", "うめ", "cây mai"), ("入梅", "にゅうばい", "bước vào mùa mưa .")]},
    "恥": {"viet": "SỈ", "meaning_vi": "Xấu hổ. Nguyễn Trãi [阮薦] : Quốc thù tẩy tận thiên niên sỉ [國讎洗盡千年恥] (Đề kiếm [題劍]) Thù nước đã rửa sạch cái nhục nghìn năm.", "meo": "", "reading": "は.じる はじ は.じらう は.ずかしい", "vocab": [("恥", "はじ", "sự xấu hổ"), ("恥じる", "はじる", "cảm thấy xấu hổ; ngượng ngùng")]},
    "応": {"viet": "ỨNG", "meaning_vi": "Đáp ứng, ứng đối,", "meo": "", "reading": "あた.る まさに こた.える", "vocab": [("一応", "いちおう", "một khi; nhất thời; tạm thời"), ("供応", "きょうおう", "sự đãi")]},
    "肪": {"viet": "PHƯƠNG", "meaning_vi": "Chi phương [脂肪] mỡ lá.", "meo": "", "reading": "ボウ", "vocab": [("脂肪", "しぼう", "mỡ ."), ("乳脂肪", "にゅうしぼう", "Chất béo trong sữa .")]},
    "坊": {"viet": "PHƯỜNG", "meaning_vi": "Phường, tên gọi các ấp các làng.", "meo": "", "reading": "ボウ ボッ", "vocab": [("坊や", "ぼうや", "con trai"), ("坊主", "この悪ガキ［わんぱく坊主］、走り回るのをやめろ:\"Đừng chạy nữa! thằng quỷ nhỏ\" ※ 始末に負えないやんちゃ坊主である:Một thằng bé nghịch ngợm bướng bỉnh khó bảo. ※ Ghi chú: cách gọi yêu con trai", "hòa thượng; tăng lữ")]},
    "防": {"viet": "PHÒNG", "meaning_vi": "Cái đê.", "meo": "", "reading": "ふせ.ぐ", "vocab": [("防ぐ", "ふせぐ", "đề phòng"), ("予防", "よぼう", "ngừa")]},
    "訪": {"viet": "PHÓNG, PHỎNG", "meaning_vi": "Tới tận nơi mà hỏi. Như thái phóng dân tục [採訪民俗] xét hỏi tục dân.", "meo": "", "reading": "おとず.れる たず.ねる と.う", "vocab": [("訪れ", "おとずれ", "sự đi thăm"), ("訪中", "ほうちゅう", "phòng bếp")]},
    "放": {"viet": "PHÓNG, PHỎNG", "meaning_vi": "Buông, thả. Như phóng ưng [放鷹] thả chim cắt ra, phóng hạc [放鶴] thả chim hạc ra, v.v.", "meo": "", "reading": "はな.す -っぱな.し はな.つ はな.れる こ.く ほう.る", "vocab": [("放す", "はなす", "buông tay; rời tay; dừng tay; thả tay; thả; buông"), ("放つ", "はなつ", "bắn")]},
    "敷": {"viet": "PHU", "meaning_vi": "Bày, mở rộng ra, ban bố khắp cả. Như phu thiết [敷設] bày biện sắp xếp.", "meo": "", "reading": "し.く -し.き", "vocab": [("敷く", "しく", "trải; lát; đệm; lắp đặt"), ("下敷", "したじき", "vật dùng để trải phía dưới; tấm kê phía dưới giấy viết; giấy kê dưới để can lên trên .")]},
    "激": {"viet": "KÍCH", "meaning_vi": "Xói, cản nước đang chảy mạnh cho nó vọt lên gọi là kích. Như kích lệ [激厲], kích dương [激揚] đều chỉ vệ sự khéo dùng người khiến cho người ta phấn phát chí khí lên cả.", "meo": "", "reading": "はげ.しい", "vocab": [("刺激", "しげき", "sự kích thích; kích thích ."), ("激励", "げきれい", "sự động viên; sự cổ vũ; sự khích lệ; sự khuyến khích; động viên; cổ vũ; khích lệ; khích lệ; khuyến khích; động viên")]},
    "抜": {"viet": "BẠT", "meaning_vi": "Rút ra", "meo": "", "reading": "ぬ.く -ぬ.く ぬ.き ぬ.ける ぬ.かす ぬ.かる", "vocab": [("抜く", "ぬく", "bạt"), ("不抜", "ふばつ", "hãng")]},
    "傷": {"viet": "THƯƠNG", "meaning_vi": "Vết đau.", "meo": "", "reading": "きず いた.む いた.める", "vocab": [("傷", "きず", "vết thương; vết xước; vết sẹo; thương tích; thương tật"), ("傷み", "いたみ", "nỗi đau buồn")]},
    "採": {"viet": "THẢI, THÁI", "meaning_vi": "Hái. Như thải liên [採蓮] hái sen, thải cúc [採菊] hái cúc, v.v.", "meo": "", "reading": "と.る", "vocab": [("採る", "とる", "chấp nhận; thừa nhận; hái (quả) ."), ("伐採", "ばっさい", "việc chặt (cây)")]},
    "彩": {"viet": "THẢI, THÁI", "meaning_vi": "Tia sáng.", "meo": "", "reading": "いろど.る", "vocab": [("彩り", "いろどり", "sự tô màu"), ("彩る", "いろどる", "nhuộm màu; nhuộm")]},
    "廊": {"viet": "LANG", "meaning_vi": "Mái hiên, hành lang.", "meo": "", "reading": "ロウ", "vocab": [("廊下", "ろうか", "gác"), ("回廊", "かいろう", "hành lang")]},
    "限": {"viet": "HẠN", "meaning_vi": "Giới hạn, cõi, có cái phạm vi nhất định không thể vượt qua được gọi là hạn. Như hạn chế [限制] nói về địa vị đã chỉ định, hạn kỳ [限期] hẹn kỳ, nói về thì giờ đã chỉ định.", "meo": "", "reading": "かぎ.る かぎ.り -かぎ.り", "vocab": [("限り", "かぎり", "giới hạn; hạn chế; hạn"), ("限る", "かぎる", "giới hạn; hạn chế; chỉ có; chỉ giới hạn ở")]},
    "眼": {"viet": "NHÃN", "meaning_vi": "Mắt.", "meo": "", "reading": "まなこ め", "vocab": [("眼", "まなこ", "con mắt; ánh mắt"), ("眼", "め", "con mắt; thị lực")]},
    "響": {"viet": "HƯỞNG", "meaning_vi": "Tiếng.", "meo": "", "reading": "ひび.く", "vocab": [("響", "ひびき", "tiếng dội"), ("n響", "エヌきょう", "dàn nhạc giao hưởng NHK")]},
    "即": {"viet": "TỨC", "meaning_vi": "Tục dùng như chữ tức [卽].", "meo": "", "reading": "つ.く つ.ける すなわ.ち", "vocab": [("即ち", "すなわち", "có nghĩa là"), ("即位", "そくい", "sự tôn lên ngôi")]},
    "節": {"viet": "TIẾT, TIỆT", "meaning_vi": "Đốt tre, đốt cây.", "meo": "", "reading": "ふし -ぶし のっと", "vocab": [("節", "せつ", "nhịp"), ("節", "ふし", "đốt")]},
    "羊": {"viet": "DƯƠNG", "meaning_vi": "Con dê.", "meo": "", "reading": "ひつじ", "vocab": [("羊", "ひつじ", "Con cừu"), ("仔羊", "こひつじ", "Cừu non .")]},
    "詳": {"viet": "TƯỜNG", "meaning_vi": "Rõ ràng, nói đủ mọi sự không thiếu tí gì. Như tường thuật [詳述] kể rõ sự việc, tường tận [詳盡] rõ hết sự việc.", "meo": "", "reading": "くわ.しい つまび.らか", "vocab": [("不詳", "ふしょう", "không rõ ràng"), ("詳報", "しょうほう", "báo cáo tường tận")]},
    "鮮": {"viet": "TIÊN, TIỂN", "meaning_vi": "Cá tươi. Ngày xưa gọi các thứ cá ba ba là tiên thực [鮮食].", "meo": "", "reading": "あざ.やか", "vocab": [("鮮度", "せんど", "độ tươi; độ tươi mới ."), ("鮮やか", "あざやか", "rực rỡ; chói lọi")]},
    "養": {"viet": "DƯỠNG, DƯỢNG", "meaning_vi": "Nuôi lớn. Như ông Mạnh Tử [孟子] nói Cẩu đắc kỳ dưỡng vô vật bất trưởng [苟得其養無物不長] nếu được nuôi tốt không vật gì không lớn.", "meo": "", "reading": "やしな.う", "vocab": [("養い", "やしない", "sự nuôi dưỡng"), ("養う", "やしなう", "dưỡng")]},
    "美": {"viet": "MĨ", "meaning_vi": "Đẹp, cái gì có vẻ đẹp khiến cho mình thấy lấy làm thích đều gọi là mĩ. Như mĩ thuật [美術].", "meo": "", "reading": "うつく.しい", "vocab": [("美", "び", "đẹp; đẹp đẽ; mỹ"), ("美事", "びじ", "rực rỡ")]},
    "較": {"viet": "GIÁC, GIẾU, GIẢO", "meaning_vi": "Cái tay xe. Hai bên chỗ tựa xe có cái gỗ đặt ngang bắt khum về đằng trước gọi là giác.", "meo": "", "reading": "くら.べる", "vocab": [("較差", "かくさ", "dãy"), ("較べる", "くらべる", "so")]},
    "拍": {"viet": "PHÁCH", "meaning_vi": "Vả, tát, vỗ. Như phách mã đề cương [拍馬提韁] giật cương quất ngựa. Nguyễn Trãi [阮廌] : Độ đầu xuân thảo lục như yên, Xuân vũ thiêm lai thủy phách thiên [渡頭春草綠如煙, 春雨添來水拍天] (Trại đầu xuân độ [寨頭春渡]) Ở bến đò đầu trại, cỏ xuân xanh như khói, Lại thêm mưa xuân, nước vỗ vào nền trời.", "meo": "", "reading": "ハク ヒョウ", "vocab": [("拍動", "はくどう", "sự đập; tiếng đạp"), ("拍子", "ひょうし", "nhịp")]},
    "縮": {"viet": "SÚC", "meaning_vi": "Thẳng. Như tự phản nhi súc [自反而縮] tự xét lại mình mà thẳng.", "meo": "", "reading": "ちぢ.む ちぢ.まる ちぢ.める ちぢ.れる ちぢ.らす", "vocab": [("縮み", "ちぢみ", "rút ngắn; co lại (vải) ."), ("縮む", "ちぢむ", "rút ngắn; co lại; thu nhỏ lại .")]},
    "泉": {"viet": "TUYỀN, TOÀN", "meaning_vi": "Suối, nguồn. Như lâm tuyền [林泉] rừng và suối, chỉ nơi ở ẩn. Tuyền đài [泉臺] nơi có suối, cũng như hoàng tuyền [黃泉] suối vàng, đều chỉ cõi chết. Âu Dương Tu [歐陽修] : Phong hồi lộ chuyển, hữu đình dực nhiên lâm ư tuyền thượng giả, Túy Ông đình dã [峰回路轉, 有亭翼然臨於泉上者, 醉翁亭也] (Túy Ông đình kí [醉翁亭記]) Núi quanh co, đường uốn khúc, có ngôi đình như dương cánh trên bờ suối, đó là đình Ông Lão Say.", "meo": "", "reading": "いずみ", "vocab": [("泉", "いずみ", "suối"), ("泉下", "せんか", "âm ty")]},
    "源": {"viet": "NGUYÊN", "meaning_vi": "Nguồn nước.", "meo": "", "reading": "みなもと", "vocab": [("源", "げん", "bản"), ("源", "みなもと", "nguồn .")]},
    "払": {"viet": "PHẤT", "meaning_vi": "Trả tiền", "meo": "", "reading": "はら.う -はら.い -ばら.い", "vocab": [("払い", "はらい", "việc trả tiền; việc phát lương; việc chi trả"), ("払う", "はらう", "bê; chuyển dời")]},
    "鉱": {"viet": "KHOÁNG", "meaning_vi": "Khai khoáng", "meo": "", "reading": "あらがね", "vocab": [("鉱化", "こうか", "khoáng hoá"), ("鉱区", "こうく", "khu khai khoáng; khu khai thác; khu mỏ; mỏ .")]},
    "拡": {"viet": "KHUẾCH", "meaning_vi": "Khuếch đại", "meo": "", "reading": "ひろ.がる ひろ.げる ひろ.める", "vocab": [("拡充", "かくじゅう", "sự mở rộng"), ("拡大", "かくだい", "sự mở rộng; sự tăng lên; sự lan rộng")]},
    "射": {"viet": "XẠ, DẠ, DỊCH", "meaning_vi": "Bắn, cho tên vào cung nỏ mà bắn ra gọi là xạ. Phàm có cái gì tống mạnh rồi bựt ra xa đều gọi là xạ. Tô Thức [蘇軾] : Thước khởi ư tiền, sử kị trục nhi xạ chi, bất hoạch [鵲起於前, 使騎逐而射之, 不獲] (Phương Sơn Tử truyện [方山子傳]) Chim khách vụt bay trước mặt, sai người cưỡi ngựa đuổi bắn, không được.", "meo": "", "reading": "い.る さ.す う.つ", "vocab": [("射す", "さす", "chích"), ("射つ", "うつ", "sự tấn công")]},
    "謝": {"viet": "TẠ", "meaning_vi": "Từ tạ. Như tạ khách [謝客] từ không tiếp khách. Xin thôi không làm quan nữa mà về gọi là tạ chánh [謝政].", "meo": "", "reading": "あやま.る", "vocab": [("謝り", "あやまり", "lời xin lỗi; lý do để xin lỗi"), ("謝る", "あやまる", "xin lỗi")]},
    "清": {"viet": "THANH", "meaning_vi": "Trong, nước không có chút cặn nào gọi là thanh, trái với trọc [濁]. Như thanh triệt [清澈] trong suốt.", "meo": "", "reading": "きよ.い きよ.まる きよ.める", "vocab": [("清い", "きよい", "quý tộc; quý phái; trong sáng; trong sạch"), ("清め", "きよめ", "sự làm sạch; làm sạch; ; lau sạch; sự trong sạch; trong sạch; tẩy uế")]},
    "精": {"viet": "TINH", "meaning_vi": "Giã gạo cho trắng tinh (gạo ngon).", "meo": "", "reading": "セイ ショウ シヤウ", "vocab": [("精", "せい", "tinh thần; linh hồn ."), ("精々", "せいぜい", "tối đa; không hơn được nữa; nhiều nhất có thể")]},
    "浄": {"viet": "TỊNH", "meaning_vi": "Thanh tịnh", "meo": "", "reading": "きよ.める きよ.い", "vocab": [("不浄", "ふじょう", "không sạch; không trong sạch; bẩn thỉu"), ("浄化", "じょうか", "việc làm sạch; sự làm sạch")]},
    "盟": {"viet": "MINH", "meaning_vi": "Thề. Giết các muông sinh đem lễ thần rồi cùng uống máu mà thề với nhau gọi là đồng minh [同盟]. Nguyễn Du [阮攸] : Trúc thạch đa tàm phụ nhĩ minh [竹石多慚負爾盟] (Tống nhân [送人]) Rất thẹn cùng trúc đá vì ta đã phụ lời thề.", "meo": "", "reading": "メイ", "vocab": [("盟主", "めいしゅ", "minh chủ ."), ("加盟", "かめい", "sự gia nhập; sự tham gia; gia nhập; tham gia")]},
    "駐": {"viet": "TRÚ", "meaning_vi": "Đóng. Xe ngựa đỗ lại nghỉ gọi là trú.", "meo": "", "reading": "チュウ", "vocab": [("駐在", "ちゅうざい", "sự cư trú; việc ở lại một địa phương (thường với mục đích công việc) ."), ("駐屯", "ちゅうとん", "sự đồn trú (quân đội) .")]},
    "往": {"viet": "VÃNG", "meaning_vi": "đi", "meo": "Nhìn hình ảnh tưởng tượng vua (vương - 王) đi hai bước ngắn (彳)", "reading": "おう", "vocab": [("往復", "おうふく", "khứ hồi"), ("往診", "おうしん", "bác sĩ đến khám tại nhà")]},
    "辛": {"viet": "TÂN", "meaning_vi": "Can tân, can thứ tám trong mười can.", "meo": "", "reading": "から.い つら.い -づら.い かのと", "vocab": [("辛い", "つらい", "đau đớn; đau xé ruột"), ("辛い", "からい", "cay")]},
    "屈": {"viet": "KHUẤT, QUẬT", "meaning_vi": "Cong, phàm sự gì cong không duỗi được đều gọi là khuất. Như lý khuất từ cùng [理屈詞窮] lẽ khuất lời cùng, bị oan ức không tỏ ra được gọi là oan khuất [冤屈], v.v.", "meo": "", "reading": "かが.む かが.める", "vocab": [("屈む", "かがむ", "stoup"), ("不屈", "ふくつ", "bất khuất")]},
    "掘": {"viet": "QUẬT", "meaning_vi": "Đào. Như quật địa [掘地] đào đất, quật tỉnh [掘井] đào giếng.", "meo": "", "reading": "ほ.る", "vocab": [("掘る", "ほる", "bới"), ("掘削", "くっさく", "sự đào; hố đào")]},
    "版": {"viet": "BẢN", "meaning_vi": "Ván, cùng nghĩa với chữ bản [板]. Như thiền bản [禪版] một tấm gỗ được các thiền sinh thời xưa sử dụng.", "meo": "", "reading": "ハン", "vocab": [("版", "はん", "bản in"), ("一版", "いちはん", "loại sách in ra loại sách xuất bản")]},
    "仮": {"viet": "GIẢ", "meaning_vi": "Giả thuyết, giả trang,", "meo": "", "reading": "かり かり-", "vocab": [("仮", "かり", "giả định; sự giả định; giả sử; cứ cho là"), ("仮に", "かりに", "giả định; giả sử; tạm thời; tạm; cứ cho là")]},
    "販": {"viet": "PHIẾN, PHÁN", "meaning_vi": "Mua rẻ bán đắt, buôn bán. Như phiến thư [販書] buôn sách", "meo": "", "reading": "ハン", "vocab": [("信販", "しんぱん", "sự vi phạm"), ("再販", "さいはん", "sự bán lại")]},
    "阪": {"viet": "PHẢN", "meaning_vi": "Cũng như chữ phản [坂].", "meo": "", "reading": "さか", "vocab": [("京阪", "けいはん", "Kyoto và Osaka"), ("阪大", "はんだい", "Trường đại học Osaka .")]},
    # "堂" → already in main DB
    "踊": {"viet": "DŨNG", "meaning_vi": "Nhảy. Như dũng dược [踊躍] nhảy nhót. Hăng hái làm việc cũng gọi là dũng dược. Giá hàng cao lên gọi là dũng quý [踊貴] cao vọt lên.", "meo": "", "reading": "おど.る", "vocab": [("踊り", "おどり", "sự nhảy múa; múa"), ("踊る", "おどる", "nhảy")]},
    "豚": {"viet": "ĐỒN, ĐỘN", "meaning_vi": "Con lợn con. Nguyễn Du [阮攸] : Sổ huề canh đạo kê đồn ngoại [數畦秔稻雞豚外] (Nhiếp Khẩu đạo trung [灄口道中]) Vài thửa lúa tám còn thêm gà lợn.", "meo": "", "reading": "ぶた", "vocab": [("豚", "ぶた", "lợn"), ("豚児", "とんじ", "lời xin lỗi; lý do để xin lỗi")]},
    "稼": {"viet": "GIÁ", "meaning_vi": "Cấy lúa.", "meo": "", "reading": "かせ.ぐ", "vocab": [("稼ぎ", "かせぎ", "tiền kiếm được"), ("稼ぐ", "かせぐ", "kiếm (tiền)")]},
    "隊": {"viet": "ĐỘI", "meaning_vi": "Đội quân, phép nhà binh quân bộ và quân pháo thủ thì cứ 126 người gọi là một đội, quân kị mã thì 56 người là một đội. Như đội ngũ [隊伍].", "meo": "", "reading": "タイ", "vocab": [("隊", "たい", "toán ."), ("一隊", "いちたい", "đảng")]},
    "塚": {"viet": "TRỦNG", "meaning_vi": "Cái mả cao. Nguyễn Du [阮攸] : Vãng sự bi thanh trủng [往事悲青塚] (Thu chí [秋至]) Chuyện cũ chạnh thương mồ cỏ xanh.", "meo": "", "reading": "つか -づか", "vocab": [("塚", "つか", "ụ; mô đất; đống ."), ("塚穴", "つかあな", "mồ")]},
    "象": {"viet": "TƯỢNG", "meaning_vi": "Con voi.", "meo": "", "reading": "かたど.る", "vocab": [("象", "しょう", "hiện tượng; hình dạng"), ("象", "ぞう", "voi")]},
    "像": {"viet": "TƯỢNG", "meaning_vi": "Hình tượng. Như tố tượng [塑像] tô tượng.", "meo": "", "reading": "ゾウ", "vocab": [("像", "ぞう", "tượng"), ("仏像", "ぶつぞう", "tượng phật")]},
    "財": {"viet": "TÀI", "meaning_vi": "Tiền của. Là một tiếng gọi tất cả các thứ như tiền nong, đồ đạc, nhà cửa, ruộng đất, hễ có giá trị đều gọi là tài sản [財產], các đồ đạc trong cửa hàng buôn đều gọi là sinh tài [生財]. Như nhân vị tài tử, điểu vị thực vong [人為財死, 鳥為食亡] người chết vì tiền của, chim chết vì miếng ăn. Ca dao Việt Nam : Chim tham ăn sa vào vòng lưới, Cá tham mồi mắc phải lưỡi câu.", "meo": "", "reading": "たから", "vocab": [("財", "ざい", "tài sản"), ("借財", "しゃくざい", "sự vay tiền; sự vay nợ; vay tiền; vay nợ .")]},
    "津": {"viet": "TÂN", "meaning_vi": "Bến. Như quan tân [關津] cửa bến, tân lương [津梁] bờ bến, đều nói về chỗ đất giao thông cần cốt cả. Vì thế kẻ cầm quyền chính ở ngôi trọng yếu đều gọi là tân yếu [津要].", "meo": "", "reading": "つ", "vocab": [("入津", "にゅうしん", "sự nhập cảng; sự vào cảng ."), ("天津", "てんしん", "Thiên Tân")]},
    "健": {"viet": "KIỆN", "meaning_vi": "Khỏe. Như dũng kiện [勇健] khỏe mạnh, kiện mã [健馬] ngựa khỏe.", "meo": "", "reading": "すこ.やか", "vocab": [("健", "けん", "sức khoẻ"), ("保健", "ほけん", "sự bảo vệ sức khỏe")]},
    "詰": {"viet": "CẬT", "meaning_vi": "Hỏi vặn. Như cùng cật [窮詰] vặn cho cùng tận, diện cật [面詰] vặn hỏi tận mặt, v.v.", "meo": "", "reading": "つ.める つ.め -づ.め つ.まる つ.む", "vocab": [("詰み", "つみ", "sự chiếu tướng; nước cờ chiếu hết"), ("詰む", "つむ", "mịn; mau; không thông; bí")]},
    "況": {"viet": "HUỐNG", "meaning_vi": "So sánh, lấy cái này mà hình dung cái kia gọi là hình huống [形況].", "meo": "", "reading": "まし.て いわ.んや おもむき", "vocab": [("不況", "ふきょう", "không vui; tiêu điều"), ("作況", "さっきょう", "chất")]},
    "党": {"viet": "ĐẢNG", "meaning_vi": "Giống Đảng Hạng [党項], tức là giống Đường Cổ giặc bây giờ, chính là chữ đảng [黨].", "meo": "", "reading": "なかま むら", "vocab": [("党", "とう", "Đảng (chính trị)"), ("一党", "いちとう", "đảng")]},
    "競": {"viet": "CẠNH", "meaning_vi": "Mạnh. Như hùng tâm cạnh khí [雄心競氣] tâm khí hùng mạnh.", "meo": "", "reading": "きそ.う せ.る くら.べる", "vocab": [("競う", "きそう", "tranh giành nhau; ganh đua"), ("競る", "せる", "ganh đua; cạnh tranh; trả giá; bỏ giá; bán đấu giá; bán hàng dạo .")]},
    "鋭": {"viet": "DUỆ, NHUỆ", "meaning_vi": "Nhọn, mũi nhọn.", "meo": "", "reading": "するど.い", "vocab": [("鋭い", "するどい", "sắc bén"), ("先鋭", "せんえい", "gốc")]},
    "脱": {"viet": "THOÁT, ĐOÁI", "meaning_vi": "Giản thể của chữ 脫", "meo": "", "reading": "ぬ.ぐ ぬ.げる", "vocab": [("脱ぐ", "ぬぐ", "cởi (quần áo"), ("脱会", "だっかい", "sự lấy lại được")]},
    "裏": {"viet": "LÍ", "meaning_vi": "Lần lót áo.", "meo": "", "reading": "うら", "vocab": [("裏", "うら", "bề trái"), ("裏に", "うらに", "giữa")]},
    "埋": {"viet": "MAI", "meaning_vi": "Chôn. Như mai táng [埋葬] chôn cất người chết.", "meo": "", "reading": "う.める う.まる う.もれる うず.める うず.まる い.ける", "vocab": [("埋伏", "まいふく", "mai phục ."), ("埋まる", "うまる", "được chôn cất; bị mai táng; bị lấp đầy; chôn; lấp")]},
    "署": {"viet": "THỰ", "meaning_vi": "Đặt. Như bộ thự [部署] đặt ra từng bộ.", "meo": "", "reading": "ショ", "vocab": [("代署", "だいしょ", "người biên chép"), ("公署", "こうしょ", "Văn phòng chính phủ .")]},
    "緒": {"viet": "TỰ", "meaning_vi": "Đầu mối sợi tơ. Gỡ tơ phải gỡ từ đầu mối, vì thế nên sự gì đã xong hẳn gọi là tựu tự [就緒] ra mối. Sự gì bối rối ngổn ngang lắm gọi là thiên đầu vạn tự [千頭萬緒] muôn đầu nghìn mối.", "meo": "", "reading": "お いとぐち", "vocab": [("緒", "お", "dây"), ("一緒", "いっしょ", "cùng")]},
    "著": {"viet": "TRỨ, TRƯỚC, TRỮ", "meaning_vi": "Sáng, rõ rệt. Như trứ danh [著名] nổi tiếng.", "meo": "", "reading": "あらわ.す いちじる.しい", "vocab": [("著す", "あらわす", "viết; xuất bản"), ("著作", "ちょさく", "tác giả")]},
    "完": {"viet": "HOÀN", "meaning_vi": "Đủ, vẹn. Cao Bá Quát [高伯适] : Y phá lạp bất hoàn [衣破笠不完] (Đạo phùng ngạ phu [道逢餓夫]) Áo rách nón không nguyên vẹn.", "meo": "", "reading": "カン", "vocab": [("完了", "かんりょう", "sự xong xuôi; sự kết thúc; sự hoàn thành; xong xuôi; kết thúc; hoàn thành"), ("完備", "かんび", "hoàn bị .")]},
    "頑": {"viet": "NGOAN", "meaning_vi": "cứng đầu, ngoan cố", "meo": "Đầu (元) mà cứ giữ khư khư (頁) thì thật là ngoan cố.", "reading": "がん", "vocab": [("頑張る", "がんばる", "cố gắng"), ("頑固", "がんこ", "ngoan cố, bảo thủ")]},
    "枯": {"viet": "KHÔ", "meaning_vi": "Khô héo. Tục gọi thân thế kẻ giàu sang là vinh [榮], kẻ nghèo hèn là khô [枯].", "meo": "", "reading": "か.れる か.らす", "vocab": [("枯らす", "からす", "làm cho héo úa; làm cho khô héo; phơi khô; để khô; để héo; tát cạn (ao hồ)"), ("枯れる", "かれる", "héo queo")]},
    "固": {"viet": "CỐ", "meaning_vi": "Bền chắc. Nguyễn Du [阮攸] : Thạch trụ ký thâm căn dũ cố [石柱既深根愈固] (Mạnh Tử từ cổ liễu [孟子祠古柳]) Trụ đá càng sâu gốc càng bền.", "meo": "", "reading": "かた.める かた.まる かた.まり かた.い", "vocab": [("固い", "かたい", "cứng nhắc; bảo thủ"), ("固さ", "かたさ", "độ cứng; sự cứng")]},
    "居": {"viet": "CƯ, KÍ", "meaning_vi": "Ở. Như yến cư [燕居]  nhàn, nghĩa là lúc ở trong nhà nhàn rỗi không có việc gì.", "meo": "", "reading": "い.る -い お.る", "vocab": [("居る", "いる", "có"), ("居る", "おる", "có; ở; sống; có mặt")]},
    "占": {"viet": "CHIÊM, CHIẾM", "meaning_vi": "Xem, coi điềm gì để biết xấu tốt gọi là chiêm.", "meo": "", "reading": "し.める うらな.う", "vocab": [("占い", "うらない", "việc tiên đoán vận mệnh; sự bói toán ."), ("占う", "うらなう", "chiêm nghiệm")]},
    "周": {"viet": "CHU", "meaning_vi": "Khắp. Như chu đáo [周到], chu chí [周至] nghĩa là trọn vẹn trước sau, không sai suyễn tí gì.", "meo": "", "reading": "まわ.り", "vocab": [("周", "ぐるり", "vùng xung quanh; quanh"), ("周り", "まわり", "vùng xung quanh; xung quanh")]},
    "召": {"viet": "TRIỆU", "meaning_vi": "mời, triệu tập", "meo": "Bộ khẩu (口) bị dao (刀) triệu tập đến", "reading": "め", "vocab": [("召す", "めす", "mời, triệu tập, dùng, ăn, uống (kính ngữ)"), ("召集", "しょうしゅう", "triệu tập")]},
    "招": {"viet": "CHIÊU, THIÊU, THIỀU", "meaning_vi": "Vẫy.", "meo": "", "reading": "まね.く", "vocab": [("招き", "まねき", "sự mời"), ("招く", "まねく", "mời; rủ .")]},
    "超": {"viet": "SIÊU", "meaning_vi": "Vượt qua, nhảy qua.", "meo": "", "reading": "こ.える こ.す", "vocab": [("超", "ちょう", "siêu"), ("超す", "こす", "làm cho vượt quá (hạn định")]},
    "昭": {"viet": "CHIÊU", "meaning_vi": "Sáng sủa, rõ rệt. Như chiêu chương [昭章] rõ rệt.", "meo": "", "reading": "ショウ", "vocab": [("昭和", "しょうわ", "Chiêu Hoà; thời kỳ Chiêu Hoà"), ("昭昭たる", "あきらあきらたる", "trong")]},
    "照": {"viet": "CHIẾU", "meaning_vi": "Soi sáng.", "meo": "", "reading": "て.る て.らす て.れる", "vocab": [("照り", "てり", "ánh sáng mặt trời"), ("照る", "てる", "chiếu sáng")]},
    "沼": {"viet": "CHIỂU", "meaning_vi": "Cái ao hình cong. Chu Văn An [朱文安] : Ngư phù cổ chiểu long hà tại [魚浮古沼龍何在] (Miết trì [鱉池]) Cá nổi trong ao xưa, rồng ở chốn nào ?", "meo": "", "reading": "ぬま", "vocab": [("沼", "ぬま", "ao; đầm ."), ("沼地", "ぬまち", "đất ao; đầm .")]},
    "丘": {"viet": "KHÂU, KHIÊU", "meaning_vi": "Cái gò, tức là đống đất nhỏ.", "meo": "", "reading": "おか", "vocab": [("丘", "おか", "quả đồi; ngọn đồi; đồi"), ("丘上", "おかうえ", "đỉnh đồi")]},
    "浜": {"viet": "BANH", "meaning_vi": "Kênh cho tàu bè đỗ.", "meo": "", "reading": "はま", "vocab": [("浜", "はま", "bãi biển"), ("海浜", "かいひん", "bờ biển; ven biển")]},
    "攻": {"viet": "CÔNG", "meaning_vi": "Đánh, vây đánh một thành ấp nào gọi là công.", "meo": "", "reading": "せ.める", "vocab": [("攻め", "せめ", "công ."), ("主攻", "しゅおさむ", "sự chuẩn y")]},
    "紅": {"viet": "HỒNG", "meaning_vi": "Đỏ hồng (sắc hồng nhạt).", "meo": "", "reading": "べに くれない あか.い", "vocab": [("紅", "くれない", "màu đỏ"), ("紅", "べに", "đỏ thẫm")]},
    "江": {"viet": "GIANG", "meaning_vi": "Sông Giang.", "meo": "", "reading": "え", "vocab": [("江", "え", "vịnh nhỏ ."), ("入江", "いりえ", "vịnh nhỏ; lạch")]},
    "尖": {"viet": "TIÊM", "meaning_vi": "Nhọn, phàm cái gì mũi nhọn đều gọi là tiêm.", "meo": "", "reading": "とが.る さき するど.い", "vocab": [("尖る", "とがる", "nhọn sắc ."), ("尖兵", "せんぺい", "tiền đội")]},
    "砂": {"viet": "SA", "meaning_vi": "Cát vàng, đá vụn, sỏi vụn gọi là sa. Xem chữ sa [沙].", "meo": "", "reading": "すな", "vocab": [("砂", "すな", "cát ."), ("砂上", "さじょう", "trên cát .")]},
    "省": {"viet": "TỈNH", "meaning_vi": "Coi xét. Thiên tử đi tuần bốn phương gọi là tỉnh phương [省方].", "meo": "", "reading": "かえり.みる はぶ.く", "vocab": [("省", "しょう", "huyện; bộ ."), ("省く", "はぶく", "loại bỏ; lược bớt")]},
    "快": {"viet": "KHOÁI", "meaning_vi": "Sướng thích. Như nhất sinh khoái hoạt [一生快活] một đời sung sướng.", "meo": "", "reading": "こころよ.い", "vocab": [("快い", "こころよい", "dễ chịu; vui lòng; hài lòng; du dương; dễ thương; thoải mái; ngon"), ("快く", "こころよく", "tiện lợi")]},
    "松": {"viet": "TÙNG", "meaning_vi": "Cây thông, thông có nhiều thứ. Như xích tùng [赤松] thông đỏ, hắc tùng [黑松] thông đen, hải tùng [海松], ngũ tu tùng [五鬚松], v.v. Cây thông đến mùa rét vẫn xanh, nên mới ví nó như người có khí tiết và người thọ. Như trinh tùng [貞松] nói người trinh tiết, kiều tùng [喬松] nói người thọ, v.v.", "meo": "", "reading": "まつ", "vocab": [("松", "まつ", "cây thông ."), ("松原", "まつばら", "cánh đồng thông .")]},
    "総": {"viet": "TỔNG", "meaning_vi": "Tục dùng như chữ tổng [總].", "meo": "", "reading": "す.べて すべ.て ふさ", "vocab": [("総", "そう", "tổng"), ("総て", "すべて", "tất cả")]},
    "共": {"viet": "CỘNG, CUNG", "meaning_vi": "Cùng, chung. Hai ông Chu Định Công [[周定公] và Triệu Mục Công [召穆公] cùng giúp vua Lệ Vương [厲王] nhà Chu Hư  trị nước, gọi là cộng hòa [共和], nghĩa là các quan cùng hòa với nhau mà làm việc. Vì thế nên bây giờ nước nào do dân cùng công cử quan lên để trị nước gọi là nước cộng hòa [共和].  Cộng, tính gộp cả các món lại làm một gọi là cộng.  Một âm là cung. Kính, cũng như chữ cung [恭].  Đủ. Như cung trương [共張] bầy đặt đủ hết mọi cái, thường dùng như chữ cung trướng [供帳].", "meo": "", "reading": "とも とも.に -ども", "vocab": [("共", "ども", "sự cùng nhau"), ("共々", "ともども", "cùng nhau")]},
    "洪": {"viet": "HỒNG", "meaning_vi": "Cả, lớn. Như hồng lượng [洪量] lượng cả, hồng phúc [洪福] phúc lớn.", "meo": "", "reading": "コウ", "vocab": [("洪大", "ひろしだい", "lớn"), ("洪水", "こうずい", "hồng thuỷ")]},
    "港": {"viet": "CẢNG", "meaning_vi": "Sông nhánh, ngành sông. Sông lớn có một dòng chảy ngang ra mà đi thuyền được, gọi là cảng.", "meo": "", "reading": "みなと", "vocab": [("港", "みなと", "cảng ."), ("港で", "みなとで", "tại cảng .")]},
    "異": {"viet": "DỊ, DI", "meaning_vi": "Khác, trái lại với tiếng cùng. Như dị vật [異物] vật khác, dị tộc [異族] họ khác, v.v.", "meo": "", "reading": "こと こと.なる け", "vocab": [("異に", "ことに", "sự khác nhau"), ("異人", "いじん", "dị nhân .")]},
    "暴": {"viet": "BẠO, BỘC", "meaning_vi": "Tàn bạo. Như tham bạo [貪暴], bạo ngược [暴虐]. Trộm giặc gọi là bạo khách [暴客], v.v.", "meo": "", "reading": "あば.く あば.れる", "vocab": [("暴く", "あばく", "vạch trần; phơi bày; bộc lộ; làm lộ"), ("乱暴", "らんぼう", "bạo loạn; hỗn láo; vô lễ; quá đáng")]},
    "爆": {"viet": "BẠO", "meaning_vi": "Nổ, bùng nổ", "meo": "Bên trái có bộ HỎA (火) rất nóng, mà gặp đủ cả NHẬT (日), CUNG (廾), THUỶ (水) bên phải thì kiểu gì cũng BẠO (nổ tung).", "reading": "ばく", "vocab": [("爆発", "ばくはつ", "Vụ nổ, sự bùng nổ"), ("爆弾", "ばくだん", "Bom")]},
    "滞": {"viet": "TRỆ", "meaning_vi": "Giản thể của chữ 滯", "meo": "", "reading": "とどこお.る", "vocab": [("滞り", "とどこおり", "sự ứ đọng; tình trạng tù hãm"), ("滞る", "とどこおる", "đọng")]},
    "掃": {"viet": "TẢO", "meaning_vi": "Quét. Như sái tảo [洒掃] vẩy nước quét nhà. Bạch Cư Dị [白居易] : Lạc diệp mãn giai hồng bất tảo [落葉滿階紅不掃] (Trường hận ca [長恨歌]) Lá rụng đỏ đầy thềm không ai quét. Tản Đà dịch thơ : Đầy thềm ai quét lá hồng thu rơi.", "meo": "", "reading": "は.く", "vocab": [("掃く", "はく", "quét; chải"), ("一掃", "いっそう", "sự quét sạch; sự tiễu trừ")]},
    "納": {"viet": "NẠP", "meaning_vi": "Vào. Như xuất nạp [出納] số ra vào.", "meo": "", "reading": "おさ.める -おさ.める おさ.まる", "vocab": [("不納", "ふのう", "sự không trả tiền"), ("納付", "のうふ", "Sự thanh toán; sự cung cấp .")]},
    "柄": {"viet": "BÍNH", "meaning_vi": "Cái chuôi, vật gì có chuôi có cán để cầm gọi là bính. Như đao bính [刀柄] chuôi dao.", "meo": "", "reading": "がら え つか", "vocab": [("柄", "え", "cán; tay cầm; móc quai"), ("柄", "がら", "mẫu; mô hình; cán")]},
    # "別" → already in main DB
    "企": {"viet": "XÍ", "meaning_vi": "Ngóng. Như vô nhâm kiều xí [無任翹企] mong ngóng khôn xiết, xí nghiệp [企業] mong ngóng cho thành nghề nghiệp, v.v.", "meo": "", "reading": "くわだ.てる たくら.む", "vocab": [("企て", "くわだて", "sơ đồ"), ("企み", "たくらみ", "âm mưu; mưu đồ")]},
    "渋": {"viet": "SÁP", "meaning_vi": "Co rút; buồn; nhăn nhó", "meo": "", "reading": "しぶ しぶ.い しぶ.る", "vocab": [("渋", "しぶ", "cành cây"), ("渋々", "しぶ々", "miễn cưỡng; bất đắc dự")]},
    "御": {"viet": "NGỰ, NHẠ, NGỮ", "meaning_vi": "Kẻ cầm cương xe.", "meo": "", "reading": "おん- お- み-", "vocab": [("御", "ご", "ngự"), ("御上", "おかみ", "sự cai trị")]},
    "視": {"viet": "THỊ", "meaning_vi": "Nhìn kĩ, coi kĩ, trông kĩ. Như thị học [視學] coi học, thị sự [視事] trông coi công việc, v.v.", "meo": "", "reading": "み.る", "vocab": [("乱視", "らんし", "loạn thị"), ("仰視", "ぎょうし", "sự tôn kính")]},
    "孫": {"viet": "TÔN, TỐN", "meaning_vi": "Cháu.", "meo": "", "reading": "まご", "vocab": [("孫", "まご", "cháu"), ("内孫", "ないそん", "Cháu .")]},
    "祈": {"viet": "KÌ", "meaning_vi": "Cầu cúng, cầu phúc. Như kì phúc [祈福] cầu phúc.", "meo": "", "reading": "いの.る", "vocab": [("祈り", "いのり", "cầu nguyện"), ("祈る", "いのる", "cầu nguyện")]},
    "株": {"viet": "CHU, CHÂU", "meaning_vi": "Gốc cây (gốc cây ở trên đất). Tống [宋] điền phủ thấy con thỏ dập đầu vào gốc cây mà chết, mới nghỉ cầy canh giữ gốc cây mong lại được thỏ đến nữa, vì thế nên những kẻ giữ chết một ý kiến của mình gọi là thủ chu đãi thỏ [守株待兔] (Hàn Phi Tử  [韓非子]).", "meo": "", "reading": "かぶ", "vocab": [("株", "かぶ", "cổ phiếu"), ("お株", "おかぶ", "sở trường; điểm mạnh")]},
    "俳": {"viet": "BÀI", "meaning_vi": "Bài ưu [俳優] phường chèo.", "meo": "", "reading": "ハイ", "vocab": [("俳", "はい", "diễn viên nam; nam diễn viên ."), ("俳人", "はいじん", "nhà thơ")]},
    "輩": {"viet": "BỐI", "meaning_vi": "Bực, lũ, bọn. Như tiền bối [前輩] bực trước, hậu bối [後輩] bọn sau, ngã bối [我輩] lũ chúng ta, nhược bối [若輩] lũ chúng bay, v.v.", "meo": "", "reading": "-ばら やから やかい ともがら", "vocab": [("輩", "ともがら", "bạn"), ("下輩", "しもともがら", "dưới")]},
    "額": {"viet": "NGẠCH", "meaning_vi": "Bộ trán, trên chỗ lông mày dưới mái tóc gọi là ngạch. Như ngạch nghiễm [額廣] trán rộng.", "meo": "", "reading": "ひたい", "vocab": [("額", "がく", "cái trán; trán (người)"), ("額", "ひたい", "trán")]},
    "閣": {"viet": "CÁC", "meaning_vi": "Gác, từng gác để chứa đồ.", "meo": "", "reading": "カク", "vocab": [("閣下", "かっか", "ngài; quý ngài"), ("倒閣", "とうかく", "sự đảo chính; sự lật đổ chính quyền")]},
    "略": {"viet": "LƯỢC", "meaning_vi": "Mưu lược, phần nhiều chỉ về việc binh. Như thao lược [韜略] có tài tháo vát. Người nào đảm đang có tài cũng gọi là thao lược. Phương lược [方略] sách chép về võ công.", "meo": "", "reading": "ほぼ おか.す おさ.める はかりごと はか.る はぶ.く りゃく.す りゃく.する", "vocab": [("略", "ほぼ", "khoảng; áng chừng; đại để"), ("略", "りゃく", "sự lược bỏ")]},
    "桜": {"viet": "ANH", "meaning_vi": "anh đào", "meo": "", "reading": "さくら", "vocab": [("桜", "さくら", "anh đào"), ("桜んぼ", "さくらんぼ", "quả anh đào .")]},
    "案": {"viet": "ÁN", "meaning_vi": "Cái bàn. Như phục án [伏案] cúi đầu trên bàn, chỉ sự chăm học, án thư [案書] bàn để sách, để đọc sách .", "meo": "", "reading": "つくえ", "vocab": [("案", "あん", "dự thảo; ý tưởng; ngân sách; đề xuất; phương án"), ("一案", "いちあん", "quan niệm")]},
    "宴": {"viet": "YẾN", "meaning_vi": "Yên nghỉ. Như tịch nhiên yến mặc [寂然宴默] yên tĩnh trầm lặng.", "meo": "", "reading": "うたげ", "vocab": [("宴", "うたげ", "đảng"), ("宴会", "えんかい", "bữa tiệc; tiệc tùng; tiệc chiêu đãi; tiệc")]},
    "刊": {"viet": "KHAN, SAN", "meaning_vi": "Chặt. Như khan mộc [刊木] chặt cây.", "meo": "", "reading": "カン", "vocab": [("休刊", "きゅうかん", "số  cũ"), ("公刊", "こうかん", "sự công bố")]},
    "軒": {"viet": "HIÊN", "meaning_vi": "Cái xe uốn hình cong mà hai bên có màn che. Lễ ngày xưa từ quan đại phu trở lên mới được đi xe ấy cho nên mới gọi người sang là hiên miện [軒冕]. Nguyễn Trãi [阮廌] : Thành trung hiên miện tổng trần sa [城中軒冕總塵沙] (Họa hữu nhân yên hà ngụ hứng [和友人煙霞寓興]) Ngựa xe, mũ áo trong thành thảy là cát bụi.", "meo": "", "reading": "のき", "vocab": [("軒", "のき", "mái chìa ."), ("一軒", "いっけん", "một căn (nhà)")]},
    "宇": {"viet": "VŨ", "meaning_vi": "Dưới mái hiên, nhà ở cũng gọi là vũ. Như quỳnh lâu ngọc vũ [瓊樓玉宇] lầu quỳnh nhà ngọc.", "meo": "", "reading": "ウ", "vocab": [("宇内", "うだい", "cả thế giới"), ("堂宇", "どうう", "công trình xây dựng lớn")]},
    "評": {"viet": "BÌNH", "meaning_vi": "Phê bình, bình phẩm, nghĩa là đem việc gì đã làm hay văn chương sách vở đã làm ra mà bàn định phải trái hay dở vậy. Hứa Thiệu [許劭] nhà Hậu Hán [後漢] hay bàn bạc các nhân vật trong làng mạc, mỗi tháng lại đổi một phẩm đề khác, gọi là nguyệt đán bình [月旦評].", "meo": "", "reading": "ヒョウ", "vocab": [("評", "ひょう", "bình luận; phê bình"), ("不評", "ふひょう", "tình trạng bị ghét bỏ")]},
    "責": {"viet": "TRÁCH, TRÁI", "meaning_vi": "Mong cầu, phận sự phải làm mà cầu cho tất phải làm cho trọn gọi là trách. Như trách nhậm [責任] phần việc mình gánh nhận, trách vọng [責望] yêu cầu kì vọng với nhau, phụ trách [負責] đảm nhận công việc.", "meo": "", "reading": "せ.める", "vocab": [("責め", "せめ", "sự khủng bố"), ("責任", "せきにん", "trách")]},
    "債": {"viet": "TRÁI", "meaning_vi": "Nợ. Như phụ trái [負債] mang nợ.", "meo": "", "reading": "サイ", "vocab": [("債", "さい", "khoản nợ; khoản vay ."), ("債主", "さいぬし", "người chủ nợ")]},
    "丁": {"viet": "ĐINH, CHÊNH, TRANH", "meaning_vi": "Can Đinh, can thứ tư trong mười can.", "meo": "", "reading": "ひのと", "vocab": [("丁", "ちょう", "bánh; khu"), ("丁", "てい", "Đinh (can) .")]},
    "頂": {"viet": "ĐÍNH", "meaning_vi": "Đỉnh đầu. Phàm chỗ nào rất cao đều gọi là đính. Như sơn đính [山頂] đỉnh núi, ốc đính [屋頂] nóc nhà, v.v.", "meo": "", "reading": "いただ.く いただき", "vocab": [("頂", "いただき", "đỉnh; chóp núi"), ("頂き", "いただき", "đường xoắn ốc")]},
    "灯": {"viet": "ĐĂNG", "meaning_vi": "Tục dùng như chữ đăng [燈].", "meo": "", "reading": "ひ ほ- ともしび とも.す あかり", "vocab": [("灯", "ともしび", "Ánh sáng"), ("灯", "ひ", "cái đèn")]},
    "庁": {"viet": "SẢNH", "meaning_vi": "Đại sảnh", "meo": "", "reading": "やくしょ", "vocab": [("庁", "ちょう", "cục"), ("官庁", "かんちょう", "cơ quan chính quyền; bộ ngành; cơ quan")]},
    "停": {"viet": "ĐÌNH", "meaning_vi": "Đứng, nửa chừng đứng lại gọi là đình. Như đình lưu [停留] dừng ở lại, đình bạc [停泊] đỗ thuyền lại, v.v.", "meo": "", "reading": "と.める と.まる", "vocab": [("停", "てい", "sự dừng"), ("停会", "ていかい", "sự hoãn lại")]},
    "誰": {"viet": "THÙY", "meaning_vi": "Gì, là tiếng nói không biết rõ tên mà hỏi. Như tính thậm danh thùy [姓甚名誰] tên họ là gì ?", "meo": "", "reading": "だれ たれ た", "vocab": [("誰", "だれ", "ai"), ("誰か", "だれか", "ai đó; một ai đó")]},
    "雑": {"viet": "TẠP", "meaning_vi": "Tạp chí, tạp kĩ", "meo": "", "reading": "まじ.える まじ.る", "vocab": [("雑", "ざつ", "sự tạp nham; tạp nham ."), ("乱雑", "らんざつ", "lẫn lộn; tạp nham; bừa bãi")]},
    "雇": {"viet": "CỐ", "meaning_vi": "Tên một giống chim.", "meo": "", "reading": "やと.う", "vocab": [("雇い", "やとい", "người làm"), ("雇う", "やとう", "thuê người; thuê người làm; thuê mướn; tuyển dụng .")]},
    "護": {"viet": "HỘ", "meaning_vi": "Giúp đỡ. Như hộ vệ [護衛], bảo hộ [保護] che chở giữ gìn, v.v.", "meo": "", "reading": "まも.る", "vocab": [("介護", "かいご", "sự chăm sóc bệnh nhân"), ("保護", "ほご", "sự bảo hộ")]},
    "権": {"viet": "QUYỀN", "meaning_vi": "Một dạng của chữ quyền [權].", "meo": "", "reading": "おもり かり はか.る", "vocab": [("権", "けん", "quyền; quyền lợi; thẩm quyền"), ("主権", "しゅけん", "chủ quyền .")]},
    "勧": {"viet": "KHUYẾN", "meaning_vi": "Khuyến cáo", "meo": "", "reading": "すす.める", "vocab": [("勧", "すすむ", "giới thiệu"), ("勧め", "すすめ", "sự giới thiệu")]},
    "催": {"viet": "THÔI", "meaning_vi": "Thúc giục. Cao Bá Quát [高伯适] : Thanh Đàm thôi biệt duệ [清潭催別袂] (Thanh Trì phiếm chu nam hạ [清池汎舟南下]) Giục giã chia tay ở Thanh Đàm.", "meo": "", "reading": "もよう.す もよお.す", "vocab": [("催し", "もよおし", "cuộc hội họp; meeting"), ("催す", "もよおす", "có triệu chứng; sắp sửa; cảm thấy")]},
    "菓": {"viet": "QUẢ", "meaning_vi": "Tục dùng như chữ quả [果].", "meo": "", "reading": "カ", "vocab": [("乳菓", "にゅうか", "Kẹo sữa ."), ("菓子", "かし", "bánh kẹo")]},
    "課": {"viet": "KHÓA", "meaning_vi": "Thi, tính. Phàm định ra khuôn phép mà thí nghiệm tra xét đều gọi là khóa. Như khảo khóa [考課] khóa thi, công khóa [工課] khóa học, v.v.", "meo": "", "reading": "カ", "vocab": [("課", "か", "bài (học)"), ("課す", "かす", "(+ on")]},
    "巣": {"viet": "SÀO", "meaning_vi": "Sào huyệt.", "meo": "", "reading": "す す.くう", "vocab": [("巣", "す", "hang ổ; sào huyệt"), ("卵巣", "らんそう", "buồng trứng")]},
    "賃": {"viet": "NHẪM", "meaning_vi": "Làm thuê.", "meo": "", "reading": "チン", "vocab": [("賃借", "ちんしゃく", "sự thuê"), ("家賃", "やちん", "tiền nhà")]},
    "妊": {"viet": "NHÂM", "meaning_vi": "mang thai, có thai", "meo": "Người PHỤ NỮ (女) đang ẩn mình (壬) vì đang mang thai.", "reading": "にん", "vocab": [("妊娠", "にんしん", "sự mang thai"), ("妊婦", "にんぷ", "người mang thai, bà bầu")]},
    "庭": {"viet": "ĐÌNH, THÍNH", "meaning_vi": "Sân trước. Nguyễn Du [阮攸] : Vô ngôn độc đối đình tiền trúc [無言獨對庭前竹] (Ký hữu [寄友]) Một mình không nói, trước khóm trúc ngoài sân. Quách Tấn dịch thơ : Lặng lẽ bên sân lòng đối trúc.", "meo": "", "reading": "にわ", "vocab": [("庭", "てい", "vườn"), ("庭", "にわ", "sân")]},
    "遊": {"viet": "DU", "meaning_vi": "Chơi, tới chỗ cảnh đẹp ngắm nghía cho thích gọi là du. Như du sơn [遊山] chơi núi, du viên [遊園] chơi vườn, v.v.", "meo": "", "reading": "あそ.ぶ あそ.ばす", "vocab": [("遊び", "あそび", "sự vui chơi; sự nô đùa"), ("遊ぶ", "あそぶ", "chơi; vui chơi; đùa giỡn")]},
    # "涼" → already in main DB
    "景": {"viet": "CẢNH", "meaning_vi": "Cảnh, cái gì hình sắc phân phối có vẻ đẹp thú đều gọi là cảnh. Như phong cảnh [風景], cảnh vật [景物] đều chỉ cảnh tượng tự nhiên trước mắt, v.v.", "meo": "", "reading": "ケイ", "vocab": [("景仰", "けいこう", "sự kính yêu"), ("光景", "こうけい", "quang cảnh; phong cảnh; cảnh vật; cảnh tượng")]},
    "影": {"viet": "ẢNH", "meaning_vi": "Bóng, cái gì có hình tất có bóng, nên sự gì có quan thiệp đến gọi là ảnh hưởng [影響].", "meo": "", "reading": "かげ", "vocab": [("影", "かげ", "bóng dáng"), ("ご影", "ごえい", "tranh thần thánh; hình ảnh của những vị đáng kính .")]},
    "僕": {"viet": "PHÓ, BỘC", "meaning_vi": "Đầy tớ.", "meo": "", "reading": "しもべ", "vocab": [("僕", "ぼく", "tôi"), ("僕ら", "ぼくら", "chúng tôi")]},
    "就": {"viet": "TỰU", "meaning_vi": "Nên. Sự đã nên gọi là sự tựu [事就].", "meo": "", "reading": "つ.く つ.ける", "vocab": [("就く", "つく", "bắt tay vào làm; bắt đầu"), ("就中", "なかんづく", "Đặc biệt là; nhất là .")]},
    "沈": {"viet": "TRẦM, THẨM, TRẤM", "meaning_vi": "Chìm. Bị chìm đắm sâu không ra ngay được gọi là trầm mê [沈迷], trầm nịch [沈溺] chìm đắm. Cũng viết là trầm [沉].", "meo": "", "reading": "しず.む しず.める", "vocab": [("沈む", "しずむ", "buồn bã; đau khổ; chìm đắm; đắm mình"), ("沈下", "ちんか", "sự lún")]},
    "傘": {"viet": "TÁN, TẢN", "meaning_vi": "Cái tán.", "meo": "", "reading": "かさ", "vocab": [("傘", "かさ", "cái ô"), ("傘下", "さんか", "dưới ô dù; sự dưới chướng; sự nép bóng")]},
    "毒": {"viet": "ĐỘC, ĐỐC", "meaning_vi": "Ác. Như độc kế [毒計] kế ác. Nguyễn Du [阮攸] : Bất lộ trảo nha dữ giác độc [不露爪牙與角毒] (Phản Chiêu hồn [反招魂]) Không để lộ ra nanh vuốt nọc độc.", "meo": "", "reading": "ドク", "vocab": [("毒", "どく", "độc hại; có hại"), ("中毒", "ちゅうどく", "nghiền; nghiện; ghiền")]},
    "慣": {"viet": "QUÁN", "meaning_vi": "Quen. Như tập quán [習慣] tập quen.", "meo": "", "reading": "な.れる な.らす", "vocab": [("慣れ", "なれ", "kinh nghiệm; thực hành"), ("慣例", "かんれい", "có tính lề thói tập quán; tập quán; thói quen")]},
    "籍": {"viet": "TỊCH, TẠ", "meaning_vi": "Sách vở, sổ sách, sách để ghi chép mọi sự cũng gọi là tịch. Như thư tịch [書籍] sách vở tài liệu. Nguyễn Du [阮攸] : Bạc mệnh hữu duyên lưu giản tịch [薄命有緣留簡籍] (Điệp tử thư trung [蝶死書中]) Mệnh bạc (nhưng) có duyên được lưu lại trong sách vở.", "meo": "", "reading": "セキ", "vocab": [("党籍", "とうせき", "Đảng tịch"), ("入籍", "にゅうせき", "nhập tịch")]},
    "逃": {"viet": "ĐÀO", "meaning_vi": "Trốn. Như đào nạn [逃難] trốn nạn, lánh nạn, đào trái [逃債] trốn nợ v.v.", "meo": "", "reading": "に.げる に.がす のが.す のが.れる", "vocab": [("逃げ", "にげ", "Sự bỏ trốn; sự bỏ chạy ."), ("逃す", "のがす", "bỏ lỡ")]},
    "渇": {"viet": "KHÁT", "meaning_vi": "khát", "meo": "Ba chấm THUỶ biến mất khi TRĂNG lên làm KHÁT nước.", "reading": "かわ", "vocab": [("渇水", "かっすい", "hạn hán"), ("渇望", "かつぼう", "khao khát")]},
    "更": {"viet": "CANH, CÁNH", "meaning_vi": "Đổi. Như canh trương [更張] đổi cách chủ trương, canh đoan [更端] đổi đầu mối khác, v.v.", "meo": "", "reading": "さら さら.に ふ.ける ふ.かす", "vocab": [("更々", "さらさら", "sự xào xạc"), ("更に", "さらに", "hơn nữa; hơn hết; trên hết")]},
    "硬": {"viet": "NGẠNH", "meaning_vi": "Cứng rắn.", "meo": "", "reading": "かた.い", "vocab": [("硬い", "かたい", "cứng; cứng rắn"), ("硬さ", "かたさ", "độ cứng; cứng rắn; rắn chắc .")]},
    "抱": {"viet": "BÃO", "meaning_vi": "Ôm, bế.", "meo": "", "reading": "だ.く いだ.く かか.える", "vocab": [("抱え", "かかえ", "ôm"), ("抱く", "だく", "bao trùm")]},
    "泡": {"viet": "PHAO, BÀO", "meaning_vi": "Bọt nước.", "meo": "", "reading": "あわ", "vocab": [("泡", "あわ", "bong bóng; bọt"), ("一泡", "ひとあわ", "cú đánh đòn")]},
    "砲": {"viet": "PHÁO", "meaning_vi": "Cũng như chữ pháo [礮] hay [炮].", "meo": "", "reading": "ホウ", "vocab": [("砲", "ほう", "súng thần công; pháo ."), ("砲丸", "ほうがん", "vỏ; bao; mai")]},
    "捕": {"viet": "BỘ", "meaning_vi": "Bắt, tới thẳng nhà kẻ có tội mà bắt gọi là đãi [逮], lùng đuổi kẻ có tội trốn là bộ [捕].", "meo": "", "reading": "と.らえる と.らわれる と.る とら.える とら.われる つか.まえる つか.まる", "vocab": [("だ捕", "だほ", "sự bắt giữ"), ("捕る", "とる", "nắm; bắt; bắt giữ")]},
    "寄": {"viet": "KÍ", "meaning_vi": "Phó thác. Như khả dĩ kí bách lí chi mệnh [可以寄百里之命] có thể phó thác cho công việc cai trị một trăm dặm được. Vì thế nên chịu gánh vác công việc phòng thủ ngoại cõi nước gọi là cương kí [疆寄].", "meo": "", "reading": "よ.る -よ.り よ.せる", "vocab": [("寄る", "よる", "ghé"), ("寄与", "きよ", "sự đóng góp")]},
    "埼": {"viet": "KỲ", "meaning_vi": "Mũi đất (nhô ra biển).", "meo": "", "reading": "さき さい みさき", "vocab": []},
    "崎": {"viet": "KHI", "meaning_vi": "Khi khu [崎嶇] đường núi gập ghềnh.", "meo": "", "reading": "さき さい みさき", "vocab": [("崎", "さき", "mũi đất (nhô ra biển) ."), ("崎崖", "きがい", "Độ dốc của ngọn núi .")]},
    "義": {"viet": "NGHĨA", "meaning_vi": "Sự phải chăng, lẽ phải chăng, nên. Định liệu sự vật hợp với lẽ phải gọi là nghĩa.", "meo": "", "reading": "ギ", "vocab": [("義", "よし", "sự công bằng"), ("一義", "いちぎ", "lý do")]},
    "司": {"viet": "TI, TƯ", "meaning_vi": "Chủ. Mỗi chức quan coi một việc gọi là ti. Như hữu ti [有司], sở ti [所司], v.v. Bây giờ các bộ đều chia riêng mỗi người giữ một việc, gọi là ti trưởng [司長].", "meo": "", "reading": "つかさど.る", "vocab": [("司る", "つかさどる", "phép tắc"), ("上司", "じょうし", "bề trên")]},
    "伺": {"viet": "TÝ, TỨ", "meaning_vi": "Dò xét, ta quen đọc là tứ.", "meo": "", "reading": "うかが.う", "vocab": [("伺い", "うかがい", "sự điều tra"), ("伺う", "うかがう", "đến thăm")]},
    "詞": {"viet": "TỪ", "meaning_vi": "Lời văn.", "meo": "", "reading": "シ", "vocab": [("冠詞", "かんし", "bài báo"), ("分詞", "ぶんし", "động tính từ")]},
    "飼": {"viet": "TỰ", "meaning_vi": "Cho ăn, chăn nuôi.", "meo": "", "reading": "か.う", "vocab": [("飼う", "かう", "chăn"), ("飼主", "かいぬし", "người nuôi các con vật; chủ nuôi")]},
    "適": {"viet": "THÍCH, ĐÍCH, QUÁT", "meaning_vi": "Đi đến. Như thích Tề [適齊] đến nước Tề.", "meo": "", "reading": "かな.う", "vocab": [("適", "てき", "giặc ."), ("適う", "かなう", "diêm")]},
    "滴": {"viet": "TÍCH, TRÍCH", "meaning_vi": "Giọt nước. Như quyên tích [涓滴] nhỏ giọt. Ta quen đọc là chữ trích. Nguyễn Trãi [阮廌] : Điểm trích sổ tàn canh [點滴數殘更] (Thính vũ [聽雨]) Điểm giọt đếm canh tàn.", "meo": "", "reading": "しずく したた.る", "vocab": [("滴", "しずく", "giọt (nước"), ("滴り", "したたり", "sự chảy nhỏ giọt")]},
    "敵": {"viet": "ĐỊCH", "meaning_vi": "Giặc thù. Như địch quốc [敵國] nước thù.", "meo": "", "reading": "かたき あだ かな.う", "vocab": [("敵", "かたき", "kẻ thù; kẻ đối đầu"), ("敵", "てき", "kẻ địch; kẻ thù")]},
    "酌": {"viet": "CHƯỚC", "meaning_vi": "Rót rượu, nay thông dụng là uống rượu. Như tiểu chước [小酌] uống xoàng, độc chước [獨酌] uống một mình.", "meo": "", "reading": "く.む", "vocab": [("お酌", "おしゃく", "gái nhảy; vũ nữ; gái chuốc rượu; gái hầu rượu"), ("参酌", "さんしゃく", "sự hỏi ý kiến")]},
    "凍": {"viet": "ĐỐNG", "meaning_vi": "Nước đông, nước đá.", "meo": "", "reading": "こお.る こご.える こご.る い.てる し.みる", "vocab": [("凍", "こお", "sự đông vì lạnh"), ("凍る", "こおる", "đặc")]},
    "乾": {"viet": "KIỀN, CAN, CÀN", "meaning_vi": "Một quẻ đầu tám quẻ (quẻ Kiền) là cái tượng lớn nhất như trời. Như vua, nên gọi tượng trời là kiền tượng [乾象], quyền vua là kiền cương [乾綱].", "meo": "", "reading": "かわ.く かわ.かす ほ.す ひ.る いぬい", "vocab": [("乾", "いぬい", "thiên đường"), ("乾き", "かわき", "làm thành khô; được dùng khô")]},
    "炎": {"viet": "VIÊM, ĐÀM, DIỄM", "meaning_vi": "Bốc cháy, ngọn lửa.", "meo": "", "reading": "ほのお", "vocab": [("炎", "ほのお", "ngọn lửa"), ("炎", "ほむら", "viêm .")]},
    "購": {"viet": "CẤU", "meaning_vi": "Mua sắm. Như cấu vật [購物] mua sắm đồ.", "meo": "", "reading": "コウ", "vocab": [("購入", "こうにゅう", "việc mua"), ("購求", "こうきゅう", "sự mua")]},
    "講": {"viet": "GIẢNG", "meaning_vi": "Hòa giải, lấy lời nói bảo cho hai bên hiểu ý tứ nhau mà hòa với nhau không tranh giành nhau nữa gọi là giảng. Như giảng hòa [講和].", "meo": "", "reading": "コウ", "vocab": [("代講", "だいこう", "người thay thế"), ("休講", "きゅうこう", "sự ngừng lên lớp; sự ngừng giảng dạy; ngừng lên lớp; nghỉ dạy")]},
    "氏": {"viet": "THỊ, CHI", "meaning_vi": "Họ, ngành họ.", "meo": "", "reading": "うじ -うじ", "vocab": [("氏", "うじ", "dòng dõi; anh (thêm vào sau tên người; ông (thêm vào sau tên người); Mr."), ("セ氏", "セし", "độ C .")]},
    "民": {"viet": "DÂN", "meaning_vi": "Người, dân, loài người thuộc ở dưới quyền chính trị gọi là dân. Như quốc dân [國民] dân nước, dân chủ [民主] chủ quyền quốc gia thuộc về toàn dân, ý nguyện nhân dân được tôn trọng theo sự tuyển cử tự do chọn người ra làm việc nước.", "meo": "", "reading": "たみ", "vocab": [("民", "たみ", "dân"), ("民主", "みんしゅ", "dân chủ; sự dân chủ .")]},
    "眠": {"viet": "MIÊN", "meaning_vi": "Ngủ, nhắm mắt. Vi Ứng Vật [韋應物] : Sơn không tùng tử lạc, U nhân ưng vị miên [山空松子落，幽人應未眠] Núi không trái tùng rụng, Người buồn chưa ngủ yên.", "meo": "", "reading": "ねむ.る ねむ.い", "vocab": [("眠い", "ねむい", "buồn ngủ"), ("眠け", "ねむけ", "Sự buồn ngủ; sự ngủ lơ mơ .")]},
    "睡": {"viet": "THỤY", "meaning_vi": "Ngủ, lúc mỏi nhắm mắt gục xuống cho tinh thần yên lặng gọi là thụy. Nguyễn Du [阮攸] : Sơn ổ hà gia đại tham thụy, Nhật cao do tự yểm sài môn [山塢何家大貪睡, 日高猶自掩柴門] (Quỷ Môn đạo trung [鬼門道中]) Trong xóm núi, nhà ai ham ngủ quá, Mặt trời đã lên cao mà cửa củi còn đóng kín. Quách Tấn dịch thơ : Nhà ai góc núi sao ham giấc, Nắng gội hiên chưa mở cánh bồng.", "meo": "", "reading": "ねむ.る ねむ.い", "vocab": [("一睡", "いっすい", "giấc ngủ; sự ngủ"), ("仮睡", "かすい", "giấc ngủ chợp")]},
    "郵": {"viet": "BƯU", "meaning_vi": "Nhà trạm. Dùng ngựa truyền tin gọi là trí [置], chạy bộ truyền tin gọi là bưu [郵].", "meo": "", "reading": "ユウ", "vocab": [("郵便", "ゆうびん", "bưu điện; dịch vụ bưu điện ."), ("郵券", "ゆうけん", "tem thư")]},
    "華": {"viet": "HOA, HÓA", "meaning_vi": "Nước Tàu. Nước Tàu tự gọi là Trung Hoa [中華], người Tàu là Hoa nhân [華人].", "meo": "", "reading": "はな", "vocab": [("中華", "ちゅうか", "Trung Hoa"), ("京華", "きょうはな", "thủ đô")]},
    "枠": {"viet": "KHUNG", "meaning_vi": "khung, viền, giới hạn", "meo": "Trong NGUYỆT (月) có vài người thợ MỘC (木) đang làm việc trong KHUNG.", "reading": "わく", "vocab": [("枠", "わく", "khung, viền"), ("窓枠", "まどわく", "khung cửa sổ")]},
    "参": {"viet": "THAM, XAM, SÂM", "meaning_vi": "Giản thể của chữ [參].", "meo": "", "reading": "まい.る まい- まじわる みつ", "vocab": [("参る", "まいる", "đi"), ("参上", "さんじょう", "sự thăm hỏi")]},
    "診": {"viet": "CHẨN", "meaning_vi": "Xem xét. Như chẩn bệnh [診病] xem bệnh, chẩn mạch [診脈] xem mạch, v.v.", "meo": "", "reading": "み.る", "vocab": [("診る", "みる", "kiểm tra; khám (thuộc y học)"), ("代診", "だいしん", "người thay thế")]},
    "珍": {"viet": "TRÂN", "meaning_vi": "Báu, đồ quý báu. Những vật gì quý báu đều gọi là trân cả. Như trân dị [珍異] quý lạ hiếm thấy.", "meo": "", "reading": "めずら.しい たから", "vocab": [("珍", "ちん", "hiếm"), ("別珍", "べっちん", "nhung vải")]},
    "修": {"viet": "TU", "meaning_vi": "Sửa, sửa cho hay tốt gọi là tu. Như tu thân [修身] sửa mình, tu đức [修德] sửa đức, tu lý cung thất [修理宮室] sửa sang nhà cửa.", "meo": "", "reading": "おさ.める おさ.まる", "vocab": [("修了", "しゅうりょう", "sự hoàn thành; sự kết thúc (khóa học) ."), ("修交", "しゅうこう", "tình hữu nghị")]},
    "鈍": {"viet": "ĐỘN", "meaning_vi": "Nhụt, đối lại với nhuệ [銳] sắc.", "meo": "", "reading": "にぶ.い にぶ.る にぶ- なま.る なまく.ら", "vocab": [("鈍", "どん", "chậm hiểu"), ("鈍い", "のろい", "chậm chạp")]},
    "純": {"viet": "THUẦN, CHUẨN, ĐỒN, TRUY", "meaning_vi": "Thành thực. Như thuần hiếu [純孝] người hiếu thực.", "meo": "", "reading": "ジュン", "vocab": [("純", "じゅん", "trong"), ("純一", "じゅんいつ", "sự sạch")]},
    "絶": {"viet": "TUYỆT", "meaning_vi": "Dị dạng của chữ [絕].", "meo": "", "reading": "た.える た.やす た.つ", "vocab": [("絶つ", "たつ", "chia tách; cắt ra; cắt đứt"), ("絶世", "ぜっせい", "có một không hai")]},
    "余": {"viet": "DƯ", "meaning_vi": "Ta (nhân xưng ngôi thứ nhất). Trần Quốc Tuấn [陳國峻] : Dư thường lâm xan vong thực, trung dạ phủ chẩm [余常臨餐忘食, 中夜撫枕] Ta thường tới bữa quên ăn, nửa đêm vỗ gối.", "meo": "", "reading": "あま.る あま.り あま.す あんま.り", "vocab": [("余", "よ", "trên; ở trên"), ("余す", "あます", "để dành; tiết kiệm; còn dư")]},
    "除": {"viet": "TRỪ", "meaning_vi": "Thềm. Như đình trừ [庭除] thềm trước sân.", "meo": "", "reading": "のぞ.く -よ.け", "vocab": [("除く", "のぞく", "giải trừ"), ("免除", "めんじょ", "sự miễn; sự miễn trừ .")]},
    "途": {"viet": "ĐỒ", "meaning_vi": "Đường lối. Như quy đồ [歸途] đường về, sĩ đồ [仕途] con đường làm quan. Nguyên là chữ đồ [涂], thông dụng chữ đồ [塗]. Cao Bá Quát [高伯适] : Cổ lai danh lợi nhân, Bôn tẩu lộ đồ trung [古來名利人，奔走路途中] (Sa hành đoản ca [沙行短歌]) Xưa nay hạng người danh lợi, Vẫn tất tả ngoài đường sá.", "meo": "", "reading": "みち", "vocab": [("途", "と", "đường"), ("一途", "いっと", "toàn tâm toàn ý; một lòng một dạ; hết lòng")]},
    "斜": {"viet": "TÀ", "meaning_vi": "Nghiêng, dốc", "meo": "Cây THẬP nằm bên cạnh BÁT cơm bị MỘT người DÙNG cán (車) NGHIÊNG đi.", "reading": "ななめ", "vocab": [("斜め", "ななめ", "Nghiêng, xiên"), ("斜線", "しゃせん", "Đường chéo, gạch chéo")]},
    "香": {"viet": "HƯƠNG", "meaning_vi": "Hơi thơm. Như hương vị [香味] hương thơm và vị ngon. Nguyễn Du [阮攸] : Thiên cổ trùng tuyền thượng hữu hương [天古重泉尙有香] (Âu Dương Văn Trung Công mộ [歐陽文忠公墓]) Nghìn thuở nơi chín suối vẫn có mùi hương.", "meo": "", "reading": "か かお.り かお.る", "vocab": [("香", "かおり", "mùi; mùi thơm; hương vị; hương; hương thơm ."), ("香り", "かおり", "hương cảng")]},
    "委": {"viet": "ỦY, UY", "meaning_vi": "Ủy thác, giao phó cho việc gì gọi là ủy. Như ủy quyền [委權] trao quyền của mình cho người khác.", "meo": "", "reading": "ゆだ.ねる", "vocab": [("委任", "いにん", "sự ủy nhiệm; ủy quyền; ủy thác"), ("委員", "いいん", "ủy viên; thành viên")]},
    "梨": {"viet": "LÊ", "meaning_vi": "Cũng như chữ lê [棃].", "meo": "", "reading": "なし", "vocab": [("梨", "なし", "quả lê"), ("梨果", "なしはて", "dạng quả táo (nạc")]},
    "勤": {"viet": "CẦN", "meaning_vi": "Siêng.", "meo": "", "reading": "つと.める -づと.め つと.まる いそ.しむ", "vocab": [("勤", "つとむ", "(thể dục"), ("勤め", "つとめ", "công việc; công vụ; nhiệm vụ; nghĩa vụ; công tác")]},
    "預": {"viet": "DỰ", "meaning_vi": "Sẵn, cùng nghĩa với chữ dự [豫]. Như dự bị [預備] phòng bị sẵn.", "meo": "", "reading": "あず.ける あず.かる", "vocab": [("預け", "あづけ", "sự coi sóc"), ("預かり", "あずかり", "sự coi sóc")]},
    "序": {"viet": "TỰ", "meaning_vi": "Hai bên tường. Hai bên giải vũ cũng gọi là lưỡng tự [兩序].", "meo": "", "reading": "つい.で ついで", "vocab": [("序", "じょ", "lời tựa; lời nói đầu"), ("序で", "ついで", "dịp; cơ hội")]},
    "柔": {"viet": "NHU", "meaning_vi": "Mềm, mềm yếu, mềm mại. Như nhu nhuyễn [柔軟] mềm lướt, nhu thuận [柔順] nhún thuận, v.v.", "meo": "", "reading": "やわ.らか やわ.らかい やわ やわ.ら", "vocab": [("優柔", "ゆうじゅう", "tính do dự"), ("柔和", "にゅうわ", "nhu hoà")]},
    "務": {"viet": "VỤ, VŨ", "meaning_vi": "Việc. Như thứ vụ [庶務] các việc.", "meo": "", "reading": "つと.める", "vocab": [("務め", "つとめ", "chức vụ; công tác"), ("事務", "じむ", "công việc")]},
    "怖": {"viet": "PHỐ, BỐ", "meaning_vi": "Hãi. Sợ hãi cuống quýt lên gọi là khủng phố [恐怖].", "meo": "", "reading": "こわ.い こわ.がる お.じる おそ.れる", "vocab": [("怖々", "こわ々", "bồn chồn"), ("怖い", "こわい", "hãi hùng")]},
    "刷": {"viet": "XOÁT, LOÁT", "meaning_vi": "Tẩy xạch.", "meo": "", "reading": "す.る -ず.り -ずり は.く", "vocab": [("刷り", "すり", "sự in"), ("印刷", "いんさつ", "dấu")]},
    "嵐": {"viet": "LAM", "meaning_vi": "Khí núi. Khí núi bốc lên nghi ngút ẩm ướt gọi là lam khí [嵐氣].", "meo": "", "reading": "あらし", "vocab": [("嵐", "あらし", "cơn bão; giông tố"), ("砂嵐", "すなあらし", "bão cát")]},
    "紫": {"viet": "TỬ", "meaning_vi": "Sắc tía, sắc tím.", "meo": "", "reading": "むらさき", "vocab": [("紫", "むらさき", "màu tím"), ("紫外", "むらさきがい", "cực tím")]},
    "従": {"viet": "TÙNG", "meaning_vi": "Phục tùng, tùy tùng, tòng thuận", "meo": "", "reading": "したが.う したが.える より", "vocab": [("従う", "したがう", "chiểu theo; căn cứ vào"), ("従事", "じゅうじ", "sự theo đuổi .")]},
    "縦": {"viet": "TÚNG", "meaning_vi": "Như chữ túng [縱].", "meo": "", "reading": "たて", "vocab": [("縦", "たて", "bề dọc"), ("縦列", "じゅうれつ", "cột")]},
    "命": {"viet": "MỆNH", "meaning_vi": "Sai khiến.", "meo": "", "reading": "いのち", "vocab": [("命", "いのち", "sinh mệnh; sự sống"), ("命", "めい", "mệnh lệnh")]},
    "印": {"viet": "ẤN", "meaning_vi": "Cái ấn (con dấu). Phép nhà Thanh định, ấn của các quan thân vương trở lên gọi là bảo [寶], từ quận vương trở xuống gọi là ấn [印], của các quan nhỏ gọi là kiêm kí [鈐記], của các quan khâm sai gọi là quan phòng [關防], của người thường dùng gọi là đồ chương [圖章] hay là tư ấn [私印].", "meo": "", "reading": "しるし -じるし しる.す", "vocab": [("印", "いん", "cái dấu"), ("印", "しるし", "dấu; dấu hiệu; biểu tượng; chứng cớ .")]},
    "層": {"viet": "TẰNG", "meaning_vi": "Từng, lớp, hai lần. Như tằng lâu [層樓] gác hai từng. Nguyễn Du [阮攸] : Thanh sơn lâu các nhất tằng tằng [青山樓閣一層層] (Thương Ngô Trúc Chi ca [蒼梧竹枝歌]) Lầu gác trên núi xanh tầng này nối tầng khác.", "meo": "", "reading": "ソウ", "vocab": [("層", "そう", "tầng lớp"), ("一層", "いっそう", "hơn nhiều; hơn một tầng; hơn một bậc")]},
    "増": {"viet": "TĂNG", "meaning_vi": "Thêm.", "meo": "", "reading": "ま.す ま.し ふ.える ふ.やす", "vocab": [("増し", "まし", "hơn; thêm; gia tăng"), ("増す", "ます", "làm tăng lên; làm hơn")]},
    "憎": {"viet": "TĂNG", "meaning_vi": "Ghét, trái lại với tiếng yêu. Như diện mục khả tăng [面目可憎] mặt mắt khá ghét. Đỗ Phủ [杜甫] : Văn chương tăng mệnh đạt, Si mị hỉ nhân qua [文章憎命達, 魑魅喜人過] (Thiên mạt hoài Lý Bạch [天末懷李白]) Văn chương ghét hạnh vận hanh thông, Yêu quái mừng khi thấy có người qua.", "meo": "", "reading": "にく.む にく.い にく.らしい にく.しみ", "vocab": [("憎い", "にくい", "đáng ghét; đáng ghê tởm; đáng yêu (với sự mỉa mai)"), ("憎さ", "にくさ", "Lòng căm ghét")]},
    "贈": {"viet": "TẶNG", "meaning_vi": "Đưa tặng. Như di tặng [遺贈] đưa tặng đồ quý, tặng thi [贈詩] tặng thơ v.v. Đỗ Phủ [杜甫] : Ưng cộng oan hồn ngữ, Đầu thi tặng Mịch La [應共冤魂語, 投詩贈汨羅] Hãy nên nói chuyện cùng hồn oan, Và gieo thơ tặng sông Mịch La (chỉ Khuất Nguyên [屈原]).", "meo": "", "reading": "おく.る", "vocab": [("贈る", "おくる", "gửi; trao cho; trao tặng; ban tặng"), ("贈与", "ぞうよ", "sự tặng; vật tặng")]},
    "震": {"viet": "CHẤN", "meaning_vi": "Sét đánh.", "meo": "", "reading": "ふる.う ふる.える", "vocab": [("震う", "ふるう", "chấn động; rung lắc ."), ("震え", "ふるえ", "run rẩy")]},
    "振": {"viet": "CHẤN, CHÂN", "meaning_vi": "Cứu giúp, cùng một nghĩa như chữ chẩn [賑].", "meo": "", "reading": "ふ.る ぶ.る ふ.り -ぶ.り ふ.るう", "vocab": [("振り", "ぶり", "phong cách; tính cách; cá tính"), ("振り", "ふり", "sự giả vờ")]},
    "娠": {"viet": "THẦN", "meaning_vi": "Chửa, đàn bà có mang đã đủ hình thể gọi là thần [娠]. Như nhâm thần [妊娠] đàn bà có mang.", "meo": "", "reading": "シン", "vocab": [("妊娠", "にんしん", "bụng phệ"), ("偽妊娠", "にせにんしん", "tính thụ thai giả")]},
    "浮": {"viet": "PHÙ", "meaning_vi": "Nổi, vật gì ở trên mặt nước gọi là phù. Nguyễn Trãi [阮廌] : Liên hoa phù thủy thượng [蓮花浮水上] (Dục Thúy sơn [浴翠山]) Hoa sen nổi trên nước.", "meo": "", "reading": "う.く う.かれる う.かぶ む う.かべる", "vocab": [("浮く", "うく", "nổi; lơ lửng"), ("浮ぶ", "うかぶ", "cái phao; phao cứu đắm")]},
    "札": {"viet": "TRÁT", "meaning_vi": "Cái thẻ, ngày xưa không có giấy, văn tự gì cũng viết vào ván gỗ nhỏ gọi là trát.", "meo": "", "reading": "ふだ", "vocab": [("札", "さつ", "tiền giấy; tờ; thẻ"), ("札", "ふだ", "thẻ; nhãn .")]},
    "宗": {"viet": "TÔNG", "meaning_vi": "Ông tông, ông tổ nhất gọi là tổ, thứ nữa là tông. Thường gọi là tông miếu, nghĩa là miếu thờ ông tổ ông tông vậy. Tục thường gọi các đời trước là tổ tông [祖宗].", "meo": "", "reading": "むね", "vocab": [("宗", "そう", "bè phái"), ("一宗", "いちむね", "bè phái")]},
    "奈": {"viet": "NẠI", "meaning_vi": "Nại hà [奈何] nài sao, sao mà. Nguyễn Trãi [阮廌] : Thần Phù hải khẩu dạ trung qua, Nại thử phong thanh nguyệt bạch hà [神符海口夜中過, 奈此風清月白何] (Quá Thần Phù hải khẩu [過神苻海口]) Giữa đêm đi qua cửa biển Thần Phù, Sao mà nơi đây gió mát trăng thanh đến thế ?", "meo": "", "reading": "いかん からなし", "vocab": [("奈何", "いかん", "thế nào"), ("奈落", "ならく", "Tận cùng; đáy; địa ngục")]},
    "整": {"viet": "CHỈNH", "meaning_vi": "Đều, ngay ngắn. Như đoan chỉnh [端整] gìn giữ quy củ nghiêm nhặt, nghiêm chỉnh [嚴整] nét mặt trang trọng, cử chỉ và dáng điệu ngay ngắn, v.v.", "meo": "", "reading": "ととの.える ととの.う", "vocab": [("整う", "ととのう", "được chuẩn bị"), ("整え", "ととのえ", "sự soạn")]},
    "証": {"viet": "CHỨNG", "meaning_vi": "Can gián. Tục mượn dùng như chữ chứng [證] nghĩa là chứng cớ.", "meo": "", "reading": "あかし", "vocab": [("証", "あかし", "Giấy chứng nhận; bằng; bằng chứng ."), ("証人", "しょうにん", "người làm chứng")]},
    "症": {"viet": "CHỨNG", "meaning_vi": "Chứng bệnh (chứng nghiệm của bệnh, gốc bệnh). Như chứng trạng [症狀] bệnh trạng.", "meo": "", "reading": "ショウ", "vocab": [("症", "しょう", "bệnh; chứng ."), ("症候", "しょうこう", "triệu chứng .")]},
    "延": {"viet": "DUYÊN", "meaning_vi": "Kéo dài. Như duyên niên [延年] thêm tuổi, duyên thọ [延壽] thêm thọ, v.v.", "meo": "", "reading": "の.びる の.べる の.べ の.ばす", "vocab": [("延び", "のび", "sự vượt quá giới hạn"), ("延べ", "のべ", "sự kéo căng; giãn dài; vuốt dài")]},
    "誕": {"viet": "ĐẢN", "meaning_vi": "Nói láo, nói toáng. Như hoang đản bất kinh [荒誕不經] láo hão không đúng sự.", "meo": "", "reading": "タン", "vocab": [("生誕", "せいたん", "sự sinh đẻ; sự ra đời ."), ("誕生", "たんじょう", "sự ra đời")]},
    "罰": {"viet": "PHẠT", "meaning_vi": "Hình phạt, phạm vào phép luật gọi là tội. Phép để trị tội gọi là hình [刑], có tội lấy hình pháp mà trị gọi là phạt [罰]. Như trừng phạt [懲罰] trị tội.", "meo": "", "reading": "ばっ.する", "vocab": [("罰", "ばち", "sự báo ứng"), ("罰", "ばつ", "sự phạt; sự trừng phạt")]},
    "荘": {"viet": "TRANG", "meaning_vi": "Trang trại.", "meo": "", "reading": "ほうき おごそ.か", "vocab": [("別荘", "べっそう", "biệt thự; nhà nghỉ"), ("荘厳", "そうごん", "sự trọng thể; sự uy nghi")]},
    "装": {"viet": "TRANG", "meaning_vi": "Giản thể của chữ [裝].", "meo": "", "reading": "よそお.う よそお.い", "vocab": [("装い", "よそおい", "quần áo"), ("装う", "よそおう", "làm dáng")]},
    "討": {"viet": "THẢO", "meaning_vi": "Đánh, đánh giết kẻ có tội gọi là thảo [討]. Như thảo tặc [討賊] đánh dẹp quân giặc.", "meo": "", "reading": "う.つ", "vocab": [("討つ", "うつ", "thảo phạt; chinh phạt"), ("討伐", "とうばつ", "sự chinh phạt .")]},
    "符": {"viet": "PHÙ", "meaning_vi": "Cái thẻ, làm bằng tre viết chữ vào rồi chẻ làm đôi, mỗi người giữ một mảnh khi nào sóng vào nhau mà đúng thì phải, là một vật để làm tin.", "meo": "", "reading": "フ", "vocab": [("符丁", "ふちょう", "đồng Mác"), ("切符", "きっぷ", "vé")]},
    "宙": {"viet": "TRỤ", "meaning_vi": "Xưa đi nay lại gọi là trụ. Như nói vũ trụ [宇宙] suốt gầm trời, vũ [宇] là nói về khoảng hư không (không gian); trụ [宙] là nói về khoảng thì giờ (thời gian) nghĩa là không gì là không bao quát hết cả ở trong đó vậy.", "meo": "", "reading": "チュウ", "vocab": [("宙", "ちゅう", "không gian"), ("宇宙", "うちゅう", "vòm trời")]},
    "届": {"viet": "GIỚI", "meaning_vi": "Tục dùng như chữ [屆].", "meo": "", "reading": "とど.ける -とど.け とど.く", "vocab": [("届", "とどけ", "giấy; đơn"), ("届く", "とどく", "chu đáo; tỉ mỉ")]},
    "制": {"viet": "CHẾ", "meaning_vi": "Phép chế, phép gì đã đặt nhất định rồi gọi là chế. Như pháp chế [法制] phép chế, chế độ [制度] thể lệ nhất định cho kẻ làm việc theo.", "meo": "", "reading": "セイ", "vocab": [("制", "せい", "chế; quy định"), ("体制", "たいせい", "thể chế .")]},
    "製": {"viet": "CHẾ", "meaning_vi": "Cắt thành áo mặc. Không học mà làm quan gọi là học chế mĩ cẩm [學製美錦].", "meo": "", "reading": "セイ", "vocab": [("製", "せい", "chế"), ("並製", "なみせい", "Sản phẩm có chất lượng trung bình .")]},
    "玄": {"viet": "HUYỀN", "meaning_vi": "Đen, sắc đen mà không có màu mỡ gọi là huyền. Như huyền hồ [玄狐] con cáo đen.", "meo": "", "reading": "ゲン", "vocab": [("玄人", "くろうと", "chuyên gia; người có chuyên môn; người có tay nghề; người lão luyện trong nghề"), ("玄冬", "けんとう", "mùa đông; đông")]},
    "畜": {"viet": "SÚC, HÚC", "meaning_vi": "Giống muông nuôi trong nhà. Như ngựa, trâu, dê, gà, chó, lợn gọi là lục súc [六畜].", "meo": "", "reading": "チク", "vocab": [("家畜", "かちく", "gia súc"), ("牧畜", "ぼくちく", "sự chăn nuôi")]},
    "過": {"viet": "QUÁ, QUA", "meaning_vi": "Vượt. Hơn. Như quá độ [過度] quá cái độ thường.", "meo": "", "reading": "す.ぎる -す.ぎる -す.ぎ す.ごす あやま.つ あやま.ち よ.ぎる", "vocab": [("過ぎ", "すぎ", "quá; hơn; sau ."), ("過ち", "あやまち", "lỗi lầm; sai lầm")]},
    "針": {"viet": "CHÂM", "meaning_vi": "Tục dùng như chữ châm [鍼].", "meo": "", "reading": "はり", "vocab": [("針", "はり", "châm"), ("偏針", "へんはり", "sự lệch")]},
    "汁": {"viet": "TRẤP, HIỆP", "meaning_vi": "Nước, nhựa. Vật gì có nước lỏng chảy ra gọi là trấp.", "meo": "", "reading": "しる -しる つゆ", "vocab": [("汁", "しる", "nước ép hoa quả; súp ."), ("乳汁", "にゅうじゅう", "Sữa; chất sữa .")]},
    "環": {"viet": "HOÀN", "meaning_vi": "Cái vòng ngọc.", "meo": "", "reading": "わ", "vocab": [("環", "たまき", "vòng ."), ("一環", "いっかん", "đuốc")]},
    "腰": {"viet": "YÊU", "meaning_vi": "Lưng. Tục gọi quả cật là yêu tử [腰子]. Nguyễn Du [阮攸] : Lục ấn triền yêu minh đắc ý [六印纏腰鳴得意] (Tô Tần đình [蘇秦亭]) Ấn tướng quốc sáu nước đeo trên lưng, được đắc ý.", "meo": "", "reading": "こし", "vocab": [("腰", "こし", "eo lưng; hông"), ("丸腰", "まるごし", "bị tước khí giới")]},
    "価": {"viet": "GIÁ", "meaning_vi": "Giá cả,  giá trị.", "meo": "", "reading": "あたい", "vocab": [("価", "あたい", "giá trị"), ("乳価", "にゅうか", "thể sữa")]},
    "煙": {"viet": "YÊN", "meaning_vi": "Khói. Như yên ba [煙波] khói sóng. Thôi Hiệu [崔顥] : Nhật mộ hương quan hà xứ thị, Yên ba giang thượng sử nhân sầu [日暮鄉關何處是, 煙波江上使人愁] (Hoàng hạc lâu [黄鶴樓]) Trời tối, quê nhà nơi đâu ? Trên sông khói sóng khiến người buồn. Tản Đà dịch thơ : Quê hương khuất bóng hoàng hôn, Trên sông khói sóng cho buồn lòng ai.", "meo": "", "reading": "けむ.る けむり けむ.い", "vocab": [("煙", "けむり", "khói"), ("煙い", "けむい", "ngạt khói; đầy khói; khói mù mịt")]},
    "票": {"viet": "PHIẾU, TIÊU, PHIÊU", "meaning_vi": "Chứng chỉ, cái dấu hiệu để nêu tên cho dễ nhận. Như hối phiếu [匯票] cái phiếu đổi lấy tiền bạc.", "meo": "", "reading": "ヒョウ", "vocab": [("票", "ひょう", "phiếu"), ("一票", "いっぴょう", "một phiếu")]},
    "標": {"viet": "TIÊU, PHIÊU", "meaning_vi": "Ngọn, đối lại với chữ bản [本]. Như tiêu bản [標本] ngọn gốc, cấp tắc trị tiêu [急則治標] kịp thì chữa cái ngọn, v.v.", "meo": "", "reading": "しるべ しるし", "vocab": [("標", "しめぎ", "đồng Mác"), ("元標", "げんぴょう", "cột mốc số không .")]},
    "騒": {"viet": "TAO", "meaning_vi": "Tao động.", "meo": "", "reading": "さわ.ぐ うれい さわ.がしい", "vocab": [("騒ぎ", "さわぎ", "sự ồn ào; sự làm ồn"), ("騒ぐ", "さわぐ", "đùa")]},
    "菌": {"viet": "KHUẨN", "meaning_vi": "Cây nấm, có thứ ăn ngon, có thứ độc lạ.", "meo": "", "reading": "キン", "vocab": [("菌", "きん", "mộng"), ("ばい菌", "ばいきん", "vi khuẩn .")]},
    "輪": {"viet": "LUÂN", "meaning_vi": "Cái bánh xe.", "meo": "", "reading": "わ", "vocab": [("輪", "わ", "bánh xe"), ("一輪", "いちりん", "bánh")]},
    "論": {"viet": "LUẬN, LUÂN", "meaning_vi": "Bàn bạc, xem xét sự vật, rồi nói cho rõ phải trái gọi là luận. Như công luận [公論] lời bàn chung của số đông người, dư luận [輿論] lời bàn của xã hội công chúng.", "meo": "", "reading": "ロン", "vocab": [("論", "ろん", "lý lẽ"), ("論う", "ろんう", "thảo luận")]},
    "輸": {"viet": "THÂU, THÚ", "meaning_vi": "Chuyển vần, lấy xe vận tải đồ đi. Như thâu tống [輸送] vận tải đưa đi, thâu xuất [輸出] vận tải ra, v.v.", "meo": "", "reading": "ユ シュ", "vocab": [("輸入", "ゆにゅう", "sự nhập khẩu ."), ("輸出", "ゆしゅつ", "sự xuất khẩu")]},
    "並": {"viet": "TỊNH, TINH", "meaning_vi": "Gồm, đều. Như tịnh lập [並立] đều đứng, tịnh hành [並行] đều đi, v.v. Có chỗ viết [竝].", "meo": "", "reading": "な.み なみ なら.べる なら.ぶ なら.びに", "vocab": [("並", "なみ", "bình thường; phổ thông"), ("並々", "なみなみ", "Bình thường .")]},
    "普": {"viet": "PHỔ", "meaning_vi": "Rộng, lớn, khắp. Như giáo dục phổ cập [教育普及] dạy dỗ khắp cả dân gian.", "meo": "", "reading": "あまね.く あまねし", "vocab": [("普", "ふ", "nói chung; đại thể ."), ("普く", "あまねく", "nhiều")]},
    "湿": {"viet": "THẤP, CHẬP", "meaning_vi": "Giản thể của chữ 濕", "meo": "", "reading": "しめ.る しめ.す うるお.う うるお.す", "vocab": [("湿す", "しめす", "làm ướt ."), ("湿り", "しめり", "sự ẩm ướt")]},
    "忙": {"viet": "MANG", "meaning_vi": "Bộn rộn, trong lòng vội gấp. Như cấp mang [急忙] vội vàng Nguyễn Du [阮攸] : Tiếu ngã bạch đầu mang bất liễu [笑我白頭忙不了] (Đông A sơn lộ hành [東阿山路行]) Cười ta đầu bạc chộn rộn chưa xong việc.", "meo": "", "reading": "いそが.しい せわ.しい おそ.れる うれえるさま", "vocab": [("多忙", "たぼう", "rất bận; bận rộn"), ("忙しい", "いそがしい", "bận")]},
    "濃": {"viet": "NÙNG", "meaning_vi": "Nồng, đặc. Trái lại với chữ đạm [淡].", "meo": "", "reading": "こ.い", "vocab": [("濃い", "こい", "có quan hệ mật thiết; gần gũi"), ("濃化", "のうか", "làm cho dày")]},
    "携": {"viet": "HUỀ", "meaning_vi": "Tục dùng như chữ huề [攜].", "meo": "", "reading": "たずさ.える たずさ.わる", "vocab": [("携帯", "けいたい", "điện thoại di động; di động"), ("必携", "ひっけい", "sổ tay .")]},
    "誘": {"viet": "DỤ", "meaning_vi": "Dỗ dành, dùng lời nói khéo khuyên người ta nghe theo mình gọi là dụ, lấy đạo nghĩa khuyên dẫn người ta làm thiện cũng gọi là dụ. Như tuần tuần thiện dụ [循循善誘] dần dần khéo dẫn dụ, nói người khéo dạy.", "meo": "", "reading": "さそ.う いざな.う", "vocab": [("誘い", "さそい", "Sự mời; sự mời mọc; sự rủ rê; mời.mời mọc; rủ rê ."), ("誘う", "さそう", "dụ")]},
    "暮": {"viet": "MỘ", "meaning_vi": "Tối, lúc mặt trời sắp lặn gọi là mộ. Nguyễn Trãi [阮廌] : Nhiễm nhiễm hàn giang khởi mộ yên 冉冉寒江起暮煙 (Thần Phù hải khẩu [神苻海口]) Trên sông lạnh khói chiều từ từ bốc lên.", "meo": "", "reading": "く.れる く.らす", "vocab": [("暮れ", "くれ", "lúc hoàng hôn; buổi chiều; cuối mùa; cuối năm"), ("暮らし", "くらし", "cuộc sống; việc sinh sống; sinh kế")]},
    "募": {"viet": "MỘ", "meaning_vi": "Tìm rộng ra. Treo một cái bảng nói rõ cách thức của mình muốn kén để cho người ta đến ứng nhận gọi là mộ. Như mộ binh [募兵] mộ lính.", "meo": "", "reading": "つの.る", "vocab": [("募る", "つのる", "chiêu mộ"), ("公募", "こうぼ", "sự tuyển dụng; sự thu hút rộng rãi; thu hút; huy động; phát hành")]},
    "似": {"viet": "TỰ", "meaning_vi": "Giống như.", "meo": "", "reading": "に.る ひ.る", "vocab": [("似る", "にる", "giống"), ("に似て", "ににて", "giống như; giống như là .")]},
    "基": {"viet": "CƠ", "meaning_vi": "Nền nhà, ở dưới cho vật gì đứng vững được đều gọi là cơ. Như căn cơ [根基] rễ cây và nền nhà, chỉ cái chính yếu để nương tựa, cơ chỉ [基址] nền móng.", "meo": "", "reading": "もと もとい", "vocab": [("基", "もと", "cơ sở; nguồn gốc; căn nguyên; gốc ban đầu"), ("基因", "もといん", "nguyên nhân")]},
    "吹": {"viet": "XUY, XÚY", "meaning_vi": "Thổi. Như xuy tiêu [吹簫] thổi tiêu, xuy địch [吹笛] thổi sáo, v.v.", "meo": "", "reading": "ふ.く", "vocab": [("吹く", "ふく", "dậy mùi"), ("息吹", "いぶき", "hơi thở")]},
    "姿": {"viet": "TƯ", "meaning_vi": "Dáng dấp thùy mị. Cho nên gọi dáng điệu con gái là tư sắc [姿色].", "meo": "", "reading": "すがた", "vocab": [("姿", "すがた", "bóng dáng"), ("姿勢", "しせい", "tư thế; điệu bộ; dáng điệu; thái độ .")]},
    "資": {"viet": "TƯ", "meaning_vi": "Của cải, vốn liếng. Như tư bản [資本] của vốn, gia tư [家資] vốn liếng nhà.", "meo": "", "reading": "シ", "vocab": [("出資", "しゅっし", "sự đầu tư; vốn đầu tư; cái được đầu tư"), ("資力", "しりょく", "tiền bạc; của cải; tiềm lực")]},
    "茨": {"viet": "TÌ", "meaning_vi": "Lợp cỏ tranh. Nguyễn Du [阮攸] : Nhất đái mao tì dương liễu trung [一帶茅茨楊柳中] (Nhiếp Khẩu đạo trung [灄口道中]) Một dãy nhà tranh trong hàng dương liễu.", "meo": "", "reading": "いばら かや くさぶき", "vocab": [("茨", "いばら", "gai")]},
    "盗": {"viet": "ĐẠO", "meaning_vi": "Giản thể của chữ 盜", "meo": "", "reading": "ぬす.む ぬす.み", "vocab": [("盗み", "ぬすみ", "Sự ăn trộm"), ("盗む", "ぬすむ", "ăn cắp")]},
    "軟": {"viet": "NHUYỄN", "meaning_vi": "Mềm. Nguyên là chữ nhuyễn [輭].", "meo": "", "reading": "やわ.らか やわ.らかい", "vocab": [("軟化", "なんか", "sự làm mềm đi ."), ("軟弱", "なんじゃく", "mềm yếu; ẻo lả; ủ rũ; nhẽo")]},
    "祖": {"viet": "TỔ", "meaning_vi": "Ông, người đẻ ra cha mình.", "meo": "", "reading": "ソ", "vocab": [("祖", "そ", "ông bà"), ("人祖", "ひとそ", "tổ tiên (người")]},
    "畳": {"viet": "ĐIỆP", "meaning_vi": "Chiếu, chiếu ngủ", "meo": "", "reading": "たた.む たたみ かさ.なる", "vocab": [("畳", "じょう", "chiếu"), ("畳", "たたみ", "chiếu .")]},
    "句": {"viet": "CÚ, CÂU, CẤU", "meaning_vi": "Câu. Hết một nhời văn gọi là nhất cú [一句] một câu.", "meo": "", "reading": "ク", "vocab": [("句", "く", "câu; ngữ; từ vựng"), ("一句", "いっく", "địa hạt")]},
    "敬": {"viet": "KÍNH", "meaning_vi": "Cung kính ngoài dáng mặt không có vẻ cợt nhợt trễ nải gọi là cung [恭], trong lòng không một chút láo lếu gọi là kính [敬]. Như kính trọng [敬重] coi người khác cao quý hơn mình.", "meo": "", "reading": "うやま.う", "vocab": [("敬い", "うやまい", "sự tôn kính; lòng sùng kính"), ("敬う", "うやまう", "tôn kính; kính trọng")]},
    "警": {"viet": "CẢNH", "meaning_vi": "Răn bảo, lấy lời nói ghê gớm khiến cho người phải chú ý nghe gọi là cảnh. Như cảnh chúng [警眾] răn bảo mọi người. Vì thế nên báo cáo những tin nguy biến ngoài biên thùy gọi là cảnh.", "meo": "", "reading": "いまし.める", "vocab": [("警め", "いましめ", "sự thận trọng"), ("警備", "けいび", "cảnh bị .")]},
    "街": {"viet": "NHAI", "meaning_vi": "Ngã tư, con đường thông cả bốn mặt, những đường cái trong thành phố đều gọi là nhai [街]. Như đại nhai tiểu hạng [大街小巷] đường lớn ngõ nhỏ, chỉ chung các nơi trong thành phố.", "meo": "", "reading": "まち", "vocab": [("街", "がい", "phố; khu"), ("街", "まち", "phố phường")]},
    "封": {"viet": "PHONG", "meaning_vi": "Phong cho, vua cho các bầy tôi đất tự trị lấy gọi là phong.", "meo": "", "reading": "フウ ホウ", "vocab": [("封", "ふう", "miệng bì thư; dấu niêm phong thư"), ("一封", "いちふう", "sự rào lại")]},
    "掛": {"viet": "QUẢI", "meaning_vi": "Treo. Như quải phàm [掛帆] treo buồm, quải niệm [掛念] lòng thắc mắc, quải hiệu [掛號] thơ bảo đảm.", "meo": "", "reading": "か.ける -か.ける か.け -か.け -が.け か.かる -か.かる -が.かる か.かり -が.かり かかり -がかり", "vocab": [("掛け", "かけ", "lòng tin; sự tín nhiệm; sự tin cậy; tín dụng"), ("掛り", "かかり", "chi phí .")]},
    "銅": {"viet": "ĐỒNG", "meaning_vi": "Đồng (Copper, Cu); một loài kim chất đỏ, ngày xưa gọi là xích kim [赤金].", "meo": "", "reading": "あかがね", "vocab": [("銅", "あかがね", "đồng"), ("銅", "どう", "đồng (kim loại) .")]},
    "筒": {"viet": "ĐỒNG", "meaning_vi": "Ống tre, ống trúc, phàm vật gì tròn mà trong có lỗ đều gọi là đồng cả. Như bút đồng [筆筒] cái tháp bút, xuy đồng [吹筒] cái ống bắn chim, v.v.", "meo": "", "reading": "つつ", "vocab": [("筒", "つつ", "ống; ống hình trụ dài"), ("筒先", "つつさき", "vòi .")]},
    "洞": {"viet": "ĐỖNG, ĐỘNG", "meaning_vi": "Cái động (hang sâu). Nguyễn Trãi [阮廌] : Thanh Hư động lí trúc thiên can [清虛洞裡竹千竿] (Mộng sơn trung [夢山中]) Trong động Thanh Hư hàng nghin cành trúc.", "meo": "", "reading": "ほら", "vocab": [("洞", "ほら", "hang; động ."), ("洞察", "どうさつ", "sự sáng suốt; sự sâu sắc; sự nhìn thấu sự việc; sự nhìn xa trông rộng")]},
    "殴": {"viet": "ẨU", "meaning_vi": "Giản thể của chữ 毆", "meo": "", "reading": "なぐ.る", "vocab": [("殴る", "なぐる", "đánh"), ("殴打", "おうだ", "đánh nhau; đánh; tấn công; ẩu đả; đánh đập")]},
    "偶": {"viet": "NGẪU", "meaning_vi": "Chợt. Như ngẫu nhiên [偶然] chợt vậy, không hẹn thế mà lại thế là ngẫu nhiên.", "meo": "", "reading": "たま", "vocab": [("偶", "ぐう", "hiếm khi; thi thoảng; hiếm thấy"), ("偶々", "たまたま", "hiếm; đôi khi; thỉnh thoảng; có lúc")]},
    "隅": {"viet": "NGUNG", "meaning_vi": "Đất ngoài ven. Như hải ngung [海隅] ngoài góc bể.", "meo": "", "reading": "すみ", "vocab": [("隅", "すみ", "góc; xó; xó xỉnh ."), ("隅々", "すみずみ", "ngóc ngách xó xỉnh; khắp nơi")]},
    "兼": {"viet": "KIÊM", "meaning_vi": "Gồm. Như kiêm quản [兼管] gồm coi, kiêm nhân [兼人] một người làm việc gồm cả việc của hai người. Tục viết là [蒹].", "meo": "", "reading": "か.ねる -か.ねる", "vocab": [("兼", "けん", "và"), ("兼任", "けんにん", "sự kiêm nhiệm; kiêm nhiệm")]},
    "嫌": {"viet": "HIỀM", "meaning_vi": "Ngờ. Cái gì hơi giống sự thực khiến cho người ngờ gọi là hiềm nghi [嫌疑].", "meo": "", "reading": "きら.う きら.い いや", "vocab": [("嫌", "いや", "khó chịu; ghét; không vừa ý"), ("嫌々", "いや々", "không bằng lòng")]},
    "刻": {"viet": "KHẮC", "meaning_vi": "Khắc, lấy dao chạm trổ vào cái gì gọi là khắc. Nguyễn Trãi [阮廌] : Bi khắc tiển hoa ban  [碑刻蘚花斑] (Dục Thúy sơn [浴翠山]) Bia khắc đã lốm đốm rêu.", "meo": "", "reading": "きざ.む きざ.み", "vocab": [("刻", "きざ", "vết xước ."), ("刻み", "きざみ", "vết khía hình V")]},
    "核": {"viet": "HẠCH", "meaning_vi": "Hạt quả. Như đào hạch [桃核] hạt đào.", "meo": "", "reading": "カク", "vocab": [("核", "かく", "hạt nhân"), ("中核", "ちゅうかく", "bộ phận nhân; lõi; trung tâm")]},
    "腹": {"viet": "PHÚC", "meaning_vi": "Bụng, dưới ngực là bụng.", "meo": "", "reading": "はら", "vocab": [("腹", "はら", "bụng"), ("お腹", "おなか", "bụng")]},
    "恵": {"viet": "HUỆ", "meaning_vi": "Ân huệ", "meo": "", "reading": "めぐ.む めぐ.み", "vocab": [("恵み", "めぐみ", "phúc lành"), ("恵む", "めぐむ", "cứu trợ; ban cho")]},
    "猫": {"viet": "MIÊU", "meaning_vi": "Tục dùng như chữ miêu [貓].", "meo": "", "reading": "ねこ", "vocab": [("猫", "ねこ", "mèo"), ("仔猫", "こねこ", "Mèo con .")]},
    "専": {"viet": "CHUYÊN", "meaning_vi": "Một dạng của chữ chuyên [專].", "meo": "", "reading": "もっぱ.ら", "vocab": [("専ら", "もっぱら", "hầu hết; chủ yếu ."), ("専任", "せんにん", "sự chuyên trách .")]},
    "博": {"viet": "BÁC", "meaning_vi": "Rộng.", "meo": "", "reading": "ハク バク", "vocab": [("博", "はく", "sự thu được; sự lấy được; sự nhận được"), ("万博", "ばんぱく", "hội chợ quốc tế .")]},
    "薄": {"viet": "BẠC, BÁC", "meaning_vi": "Cỏ mọc từng bụi gọi là bạc. Như lâm bạc [林薄] rừng rậm.", "meo": "", "reading": "うす.い うす- -うす うす.める うす.まる うす.らぐ うす.ら- うす.れる すすき", "vocab": [("薄々", "うすうす", "mỏng; mong manh"), ("薄い", "うすい", "lạt")]},
    "簿": {"viet": "BỘ, BẠC", "meaning_vi": "Sổ sách. Phàm những sách vở đặt ra để tùy thời ghi chép các sự vật đều gọi là bộ.", "meo": "", "reading": "ボ", "vocab": [("原簿", "げんぼ", "sổ cái"), ("名簿", "めいぼ", "danh bạ .")]},
    "療": {"viet": "LIỆU", "meaning_vi": "Chữa bệnh. Như trị liệu [治療] chữa bệnh.", "meo": "", "reading": "リョウ", "vocab": [("医療", "いりょう", "sự chữa trị"), ("施療", "せりょう", "sự trị liệu miễn phí .")]},
    "僚": {"viet": "LIÊU", "meaning_vi": "Đồng liêu, bạn đồng nghiệp", "meo": "Có người (亻) ở trên tòa nhà (寮) là đồng liêu.", "reading": "りょう", "vocab": [("同僚", "どうりょう", "Đồng nghiệp"), ("官僚", "かんりょう", "Quan liêu")]},
    "寮": {"viet": "LIÊU", "meaning_vi": "Cái cửa sổ nhỏ.", "meo": "", "reading": "リョウ", "vocab": [("寮", "りょう", "ký túc sinh viên; nhà ở của công nhân"), ("寮生", "りょうせい", "học sinh nội trú")]},
    "滝": {"viet": "LANG", "meaning_vi": "Thác nước", "meo": "", "reading": "たき", "vocab": [("滝", "たき", "thác nước"), ("滝口", "たきぐち", "đỉnh thác .")]},
    "際": {"viet": "TẾ", "meaning_vi": "Giao tiếp. Người ta cùng đi lại chơi bời với nhau gọi là giao tế [交際].", "meo": "", "reading": "きわ -ぎわ", "vocab": [("際", "きわ", "rìa; gờ; bờ; ven"), ("際", "さい", "dịp này; lần này")]},
    "察": {"viet": "SÁT", "meaning_vi": "Xét lại.", "meo": "", "reading": "サツ", "vocab": [("察", "さっ", "cảnh sát"), ("察し", "さっし", "sự cân nhắc")]},
    "甘": {"viet": "CAM", "meaning_vi": "Ngọt.", "meo": "", "reading": "あま.い あま.える あま.やかす うま.い", "vocab": [("甘い", "あまい", "ngon ngọt"), ("甘み", "あまみ", "tính chất ngọt")]},
    "紺": {"viet": "CÁM", "meaning_vi": "Xanh biếc, tục gọi là màu thiên thanh, mùi xanh sẫm ánh đỏ.", "meo": "", "reading": "コン", "vocab": [("紺", "こん", "màu xanh sẫm; màu xanh nước biển ."), ("紺屋", "こんや", "hàng nhuộm .")]},
    "胸": {"viet": "HUNG", "meaning_vi": "Ngực.", "meo": "", "reading": "むね むな-", "vocab": [("胸", "むね", "ngực"), ("胸中", "きょうちゅう", "trong lòng; tâm trí; nỗi niềm; nỗi lòng")]},
    "脳": {"viet": "NÃO", "meaning_vi": "Bộ não, đầu não", "meo": "", "reading": "のうずる", "vocab": [("脳", "のう", "não"), ("主脳", "しゅのう", "cái đầu (người")]},
    "券": {"viet": "KHOÁN", "meaning_vi": "Khoán, tức như cái giấy hợp đồng bây giờ, mỗi bên giữ một cái giấy để làm bằng cứ. Phàm văn tự để làm tin đều gọi là khoán.", "meo": "", "reading": "ケン", "vocab": [("券", "けん", "bản"), ("債券", "さいけん", "trái phiếu; giấy nợ; phiếu nợ")]},
    "巻": {"viet": "QUYỂN", "meaning_vi": "Quyển sách.", "meo": "", "reading": "ま.く まき ま.き", "vocab": [("巻", "まき", "cuộn ."), ("巻く", "まく", "bện")]},
    "皆": {"viet": "GIAI", "meaning_vi": "Đều, cùng, lời nói tóm cả mọi cái mọi sự.", "meo": "", "reading": "みな みんな", "vocab": [("皆", "みな", "hết thảy"), ("皆", "みんな", "mọi người")]},
    "混": {"viet": "HỖN, CỔN", "meaning_vi": "Hỗn tạp. Làm gian dối khiến cho người khó phân biệt được gọi là tệ hỗn [弊混].", "meo": "", "reading": "ま.じる -ま.じり ま.ざる ま.ぜる こ.む", "vocab": [("混む", "こむ", "đông đúc ."), ("混乱", "こんらん", "hỗn độn")]},
    "筋": {"viet": "CÂN", "meaning_vi": "Gân sức. Những thớ ở trong thịt giúp cho sức thịt co ruỗi mạnh mẽ đều gọi là cân cả. Người già yếu ớt gọi là cân cốt tựu suy [筋骨就衰].", "meo": "", "reading": "すじ", "vocab": [("筋", "すじ", "cốt truyện"), ("一筋", "ひとすじ", "dây")]},
    "励": {"viet": "LỆ", "meaning_vi": "Giản thể của chữ 勵", "meo": "", "reading": "はげ.む はげ.ます", "vocab": [("励み", "はげみ", "sự kích thích; tác dụng kích khích"), ("励む", "はげむ", "cố gắng; phấn đấu")]},
    "栃": {"viet": "(DẺ)", "meaning_vi": "Một loại hạt dẻ.", "meo": "", "reading": "とち", "vocab": []},
    "協": {"viet": "HIỆP", "meaning_vi": "Hòa hợp. Như đồng tâm hiệp lực [同心協力], hiệp thương [協商] cùng bàn để định lấy một phép nhất định.", "meo": "", "reading": "キョウ", "vocab": [("協", "きょう", "hiệp (hội)"), ("協会", "きょうかい", "dặn")]},
    "戻": {"viet": "LỆ", "meaning_vi": "Quay lại", "meo": "", "reading": "もど.す もど.る", "vocab": [("戻し", "もどし", "sự điều hướng lại"), ("戻す", "もどす", "hoàn lại; trả lại; khôi phục lại")]},
    "営": {"viet": "DOANH, DINH", "meaning_vi": "Doanh nghiệp, kinh doanh.", "meo": "", "reading": "いとな.む いとな.み", "vocab": [("営々", "えいえい", "cứng"), ("営み", "いとなみ", "sự làm việc; việc")]},
    "援": {"viet": "VIỆN", "meaning_vi": "viện trợ, giúp đỡ", "meo": "Tay (扌) đội vương miện (冖) giúp bạn đang gặp nạn (爰)", "reading": "えん", "vocab": [("応援", "おうえん", "cổ vũ, ủng hộ"), ("支援", "しえん", "hỗ trợ")]},
    "媛": {"viet": "VIỆN, VIÊN", "meaning_vi": "Con gái đẹp, có khi đọc là viên.", "meo": "", "reading": "ひめ", "vocab": [("媛", "ひめ", "bà chúa; bà hoàng; công chúa")]},
    "肌": {"viet": "CƠ", "meaning_vi": "Da. Như cơ nhục [肌肉] da thịt.", "meo": "", "reading": "はだ", "vocab": [("肌", "はだ", "bề mặt"), ("人肌", "ひとはだ", "Da; sức nóng thân thể")]},
    "処": {"viet": "XỨ", "meaning_vi": "Nguyên là chữ xứ [處].", "meo": "", "reading": "ところ -こ お.る", "vocab": [("処世", "しょせい", "hạnh kiểm"), ("何処", "どこ", "ở đâu; ở chỗ nào .")]},
    "冗": {"viet": "NHŨNG", "meaning_vi": "Cũng như chữ nhũng [宂].", "meo": "", "reading": "ジョウ", "vocab": [("冗", "じょう", "thừa; không cần thiết ."), ("冗員", "じょういん", "nhân viên dư thừa")]},
    "航": {"viet": "HÀNG", "meaning_vi": "Thuyền, hai chiếc thuyền cùng sang gọi là hàng.", "meo": "", "reading": "コウ", "vocab": [("内航", "ないこう", "Đường cảng trong nước"), ("出航", "しゅっこう", "sự rời khỏi")]},
    "恐": {"viet": "KHỦNG, KHÚNG", "meaning_vi": "Sợ. Như khủng khiếp [恐怯] rất sợ hãi.", "meo": "", "reading": "おそ.れる おそ.る おそ.ろしい こわ.い こわ.がる", "vocab": [("恐い", "こわい", "làm sợ hãi"), ("恐れ", "おそれ", "ngại")]},
    "状": {"viet": "TRẠNG", "meaning_vi": "Giản thể của chữ 狀", "meo": "", "reading": "ジョウ", "vocab": [("状", "じょう", "giấy (mời"), ("万状", "まんじょう", "sự làm cho thành nhiều dạng")]},
    "牧": {"viet": "MỤC", "meaning_vi": "Kẻ chăn giống muông. Nguyễn Trãi [阮廌] : Mục địch nhất thanh thiên nguyệt cao [牧笛一聲天月高] (Chu trung ngẫu thành [舟中偶成]) Sáo mục đồng (trổi lên) một tiếng, trăng trời cao.", "meo": "", "reading": "まき", "vocab": [("牧人", "ぼくじん", "người chăn cừu"), ("牧場", "まきば", "đồng cỏ .")]},
    "革": {"viet": "CÁCH, CỨC", "meaning_vi": "Da, da giống thú thuộc bỏ sạch lông đi gọi là cách.", "meo": "", "reading": "かわ", "vocab": [("革", "かわ", "da"), ("革命", "かくめい", "cách mạng; cuộc cách mạng")]},
    "加": {"viet": "GIA", "meaning_vi": "Thêm.", "meo": "", "reading": "くわ.える くわ.わる", "vocab": [("加", "か", "tính cộng; phép cộng; sự cộng lại"), ("付加", "ふか", "phụ thêm")]},
    "床": {"viet": "SÀNG", "meaning_vi": "Cũng như chữ sàng [牀].", "meo": "", "reading": "とこ ゆか", "vocab": [("床", "とこ", "giường"), ("床", "ゆか", "nền nhà")]},
    "替": {"viet": "THẾ", "meaning_vi": "Bỏ. Như thế phế [替廢] bỏ phế.", "meo": "", "reading": "か.える か.え- か.わる", "vocab": [("替え", "かえ", "sự đổi"), ("両替", "りょうか", "đổi tiền")]},
    "因": {"viet": "NHÂN", "meaning_vi": "Nhưng, vẫn thế.", "meo": "", "reading": "よ.る ちな.む", "vocab": [("因", "いん", "nguyên nhân"), ("因る", "よる", "nguyên do; vì")]},
    "帽": {"viet": "MẠO", "meaning_vi": "Cái mũ, các thứ dùng để đội đầu đều gọi là mạo.", "meo": "", "reading": "ずきん おお.う", "vocab": [("制帽", "せいぼう", "mũ đi học ."), ("帽子", "ぼうし", "mũ; nón")]},
    "致": {"viet": "TRÍ", "meaning_vi": "Suy cùng. Như cách trí [格致] suy cùng lẽ vật. Nghiên cứu cho biết hết thảy các vật có hình, vô hình trong khoảng trời đất, nó sinh, nó diệt, nó hợp, nó ly thế nào gọi là cách trí.", "meo": "", "reading": "いた.す", "vocab": [("致す", "いたす", "làm; xin được làm"), ("一致", "いっち", "sự nhất trí; sự giống nhau; sự thống nhất")]},
    "到": {"viet": "ĐÁO", "meaning_vi": "Đến nơi. Như đáo gia [到家] về đến nhà.", "meo": "", "reading": "いた.る", "vocab": [("到る", "いたる", "đến tới"), ("周到", "しゅうとう", "cực kỳ cẩn thận; cực kỳ kỹ lưỡng; tỉ mỉ; rất chú ý đến tiểu tiết")]},
    "握": {"viet": "ÁC", "meaning_vi": "Cầm, nắm. Nguyễn Du [阮攸] : Ác phát kinh hoài mạt nhật tâm [握髮驚懷末日心] (Thu dạ [秋夜]) Vắt tóc nghĩ mà sợ cho chí nguyện mình trong những ngày chót. Theo điển ác phát thổ bộ [握髮吐哺] vắt tóc nhả cơm : Chu Công [周公] là một đại thần của nhà Chu rất chăm lo việc nước. Đương ăn cơm, có khách đến nhả cơm ra tiếp. Đương gội đầu có sĩ phu tới, liền vắt tóc ra đón, hết người này đến người khác, ba lần mới gội đầu xong. Câu thơ của Nguyễn Du ý nói : Chí nguyện được đem ra giúp nước như Chu Công, cuối cùng không biết có toại nguyện hay chăng. Nghĩ đến lòng vô cùng lo ngại.", "meo": "", "reading": "にぎ.る", "vocab": [("握り", "にぎり", "rãnh nhỏ"), ("握る", "にぎる", "bắt")]},
    "侵": {"viet": "XÂM", "meaning_vi": "Tiến dần. Như xâm tầm [侵尋] dần dà.", "meo": "", "reading": "おか.す", "vocab": [("侵す", "おかす", "xâm nhập; vi phạm; tấn công"), ("侵入", "しんにゅう", "sự xâm nhập; sự xâm lược; xâm nhập; xâm lược .")]},
    "害": {"viet": "HẠI, HẠT", "meaning_vi": "Hại. Như di hại vô cùng [貽害無窮] để hại không cùng.", "meo": "", "reading": "ガイ", "vocab": [("害", "がい", "hại; cái hại"), ("侵害", "しんがい", "sự vi phạm; sự xâm hại; sự xâm phạm")]},
    "割": {"viet": "CÁT", "meaning_vi": "Cắt đứt. Như tâm như đao cát [心如刀割] lòng như dao cắt.", "meo": "", "reading": "わ.る わり わ.り わ.れる さ.く", "vocab": [("割", "かつ", "sự phân chia; sự chia cắt; tỷ lệ; phần"), ("割", "わり", "tỉ lệ; tỉ lệ phần trăm; đơn vị 10% .")]},
    "善": {"viet": "THIỆN, THIẾN", "meaning_vi": "Thiện, lành, đối lại với chữ ác [惡].", "meo": "", "reading": "よ.い い.い よ.く よし.とする", "vocab": [("善", "ぜん", "sự tốt đẹp; sự hoàn thiện; sự đúng đắn ."), ("善い", "よい", "tốt đẹp; hoàn thiện; vừa lòng")]},
    "述": {"viet": "THUẬT", "meaning_vi": "Bày ra, thuật ra. Chép các điều đã nghe từ trước ra gọi là thuật.", "meo": "", "reading": "の.べる", "vocab": [("上述", "じょうじゅつ", "việc đã nói ở trước"), ("述作", "じゅっさく", "mẹo")]},
    "訓": {"viet": "HUẤN", "meaning_vi": "Dạy dỗ.", "meo": "", "reading": "おし.える よ.む くん.ずる", "vocab": [("訓令", "くんれい", "dụ"), ("内訓", "ないくん", "Mệnh lệnh bí mật của cấp trên .")]},
    "頼": {"viet": "LẠI", "meaning_vi": "ỷ lại", "meo": "", "reading": "たの.む たの.もしい たよ.る", "vocab": [("頼み", "たのみ", "sự yêu cầu; sự đề nghị; sự mong muốn; sự nhờ cậy"), ("頼む", "たのむ", "cậy")]},
    "瀬": {"viet": "LẠI", "meaning_vi": "Thác nước.", "meo": "", "reading": "せ", "vocab": [("瀬", "せ", "chỗ nông; chỗ cạn"), ("川瀬", "かわせ", "Thác ghềnh .")]},
    "刺": {"viet": "THỨ, THÍCH", "meaning_vi": "Đâm chết, lấy dao đâm giết. Kẻ giết người gọi là thứ khách [刺客]. Ta quen đọc là chữ thích.", "meo": "", "reading": "さ.す さ.さる さ.し さし とげ", "vocab": [("刺", "とげ", "gai"), ("刺々", "とげ々", "sự làm phát cáu")]},
    "策": {"viet": "SÁCH", "meaning_vi": "Thẻ gấp. Ngày xưa không có giấy, việc nhỏ biên vào cái thẻ một gọi là giản [簡], việc to biên vào cái thẻ ken từng mảng to gấp lại được gọi là sách [策].", "meo": "", "reading": "サク", "vocab": [("策", "さく", "sách; sách lược; kế sách ."), ("一策", "いっさく", "quan niệm")]},
    "脈": {"viet": "MẠCH", "meaning_vi": "Mạch máu, mạch máu đỏ gọi là động mạch [動脈], mạch máu đen gọi là tĩnh mạch [靜脈]. Ngày xưa viết là [衇]. Tục viết là [脉].", "meo": "", "reading": "すじ", "vocab": [("脈", "みゃく", "mạch; nhịp đập; nhịp"), ("一脈", "いちみゃく", "tĩnh mạch")]},
    "井": {"viet": "TỈNH", "meaning_vi": "Giếng, đào sâu lấy mạch nước dùng gọi là tỉnh.", "meo": "", "reading": "い", "vocab": [("井", "い", "(+ up"), ("伊井", "いい", "tốt")]},
    "囲": {"viet": "VI", "meaning_vi": "Chu vi, bao vây", "meo": "", "reading": "かこ.む かこ.う かこ.い", "vocab": [("囲い", "かこい", "tường vây; hàng rào"), ("囲う", "かこう", "bao vây; vây; bủa vây; quây")]},
    "丼": {"viet": "", "meaning_vi": "bowl, bowl of food", "meo": "", "reading": "どんぶり", "vocab": [("丼", "どんぶり", "bát sứ; bát cơm đầy thức ăn ."), ("天丼", "てんどん", "bát cơm có cá rán .")]},
    "跡": {"viet": "TÍCH", "meaning_vi": "Vết chân. Như tung tích [蹤跡] dấu vết. Nguyễn Trãi [阮薦] : Tâm như dã hạc phi thiên tế, Tích tự chinh hồng đạp tuyết sa [心如野鶴飛天際, 跡似征鴻踏雪沙] (Họa hữu nhân yên hà ngụ hứng [和友人煙霞寓興]) Lòng như hạc nội bay giữa trời, Dấu tựa cánh chim hồng dẫm trên bãi tuyết.", "meo": "", "reading": "あと", "vocab": [("跡", "あと", "dấu vết; vết tích"), ("跡", "せき", "tích")]},
    "湾": {"viet": "LOAN", "meaning_vi": "Giản thể của chữ 灣", "meo": "", "reading": "いりえ", "vocab": [("湾", "わん", "vịnh ."), ("湾入", "わんにゅう", "vịnh; vũng")]},
    "依": {"viet": "Y", "meaning_vi": "Nhờ cậy, dựa vào", "meo": "Người (亻) mặc yếm (衣) đi nhờ vả.", "reading": "い", "vocab": [("依頼", "いらい", "Yêu cầu, thỉnh cầu"), ("依存", "いぞん", "Phụ thuộc, lệ thuộc")]},
    "崩": {"viet": "BĂNG", "meaning_vi": "Lở, núi sạt gọi là băng. Nguyễn Du [阮攸] : Băng nhai quái thạch nộ tương hướng [崩崖怪石怒相向] (Chu hành tức sự [舟行即事]) Bờ núi lở, đá hình quái dị giận dữ nhìn nhau.", "meo": "", "reading": "くず.れる -くず.れ くず.す", "vocab": [("崩す", "くずす", "phá hủy; kéo đổ; làm rối loạn"), ("崩れ", "くずれ", "đổ")]},
    "沢": {"viet": "TRẠCH", "meaning_vi": "Đầm lầy", "meo": "", "reading": "さわ うるお.い うるお.す つや", "vocab": [("沢", "さわ", "đầm nước"), ("余沢", "よたく", "sự phế truất ; sự hạ bệ")]},
    "岡": {"viet": "CƯƠNG", "meaning_vi": "Sườn núi.", "meo": "", "reading": "おか", "vocab": [("岡", "おか", "đồi"), ("岡引", "おかっぴき", "để dò ra")]},
    "群": {"viet": "QUẦN", "meaning_vi": "Cũng như chữ quần [羣].", "meo": "", "reading": "む.れる む.れ むら むら.がる", "vocab": [("群", "ぐん", "huyện; quần thể; nhóm; đàn; lũ"), ("群れ", "むれ", "tốp; nhóm; bầy đàn")]},
    "厳": {"viet": "NGHIÊM", "meaning_vi": "Tôn nghiêm, nghiêm khắc, nghiêm trọng", "meo": "", "reading": "おごそ.か きび.しい いか.めしい いつくし", "vocab": [("厳い", "いむい", "dữ tợn"), ("厳か", "おごそか", "sự uy nghiêm; sự tráng lệ; sự oai nghiêm; sự đường bệ; sự trang trọng; sự trầm hùng")]},
    "極": {"viet": "CỰC", "meaning_vi": "Cái nóc nhà.", "meo": "", "reading": "きわ.める きわ.まる きわ.まり きわ.み き.める -ぎ.め き.まる", "vocab": [("極", "ごく", "rất; vô cùng; cực; cực kỳ"), ("n極", "Ｎきょく", "cực Bắc")]},
    "与": {"viet": "DỮ, DỰ, DƯ", "meaning_vi": "Tục dùng như chữ [與].", "meo": "", "reading": "あた.える あずか.る くみ.する ともに", "vocab": [("与え", "あたえ", "sự ban cho"), ("付与", "ふよ", "sự cho")]},
    "汚": {"viet": "Ô", "meaning_vi": "Như chữ ô [汙].", "meo": "", "reading": "けが.す けが.れる けが.らわしい よご.す よご.れる きたな.い", "vocab": [("汚い", "きたない", "bẩn; ô uế; bẩn thỉu"), ("汚す", "けがす", "làm bẩn; bôi nhọ; làm hoen ố; làm nhục; xâm hại; cưỡng dâm; vấy bẩn lên")]},
    "互": {"viet": "HỖ", "meaning_vi": "Đắp đổi, hai bên cùng thay đổi với nhau. Như hỗ trợ [互助] giúp đỡ lẫn nhau.", "meo": "", "reading": "たが.い かたみ.に", "vocab": [("互い", "たがい", "cả hai bên; song phương ."), ("互に", "かたみに", "lẫn nhau")]},
    "宛": {"viet": "UYỂN, UYÊN", "meaning_vi": "Uyển nhiên [宛然] y nhiên (rõ thế).", "meo": "", "reading": "あ.てる -あて -づつ あたか.も", "vocab": [("宛", "あて", "nơi đến; nơi gửi đến"), ("宛い", "あてい", "sự phân công")]},
    "腕": {"viet": "OẢN, UYỂN", "meaning_vi": "Cổ tay. Người ta tới lúc thất ý thường chống tay thở dài, nên toan tính không ra gọi là ách oản [扼腕]. Cũng đọc là uyển.", "meo": "", "reading": "うで", "vocab": [("腕", "うで", "cánh tay"), ("凄腕", "すごうで", "(từ Mỹ")]},
    "挙": {"viet": "CỬ", "meaning_vi": "Tuyển cử, cử động, cử hành", "meo": "", "reading": "あ.げる あ.がる こぞ.る", "vocab": [("偉挙", "えらきょ", "sự phụ thuộc"), ("挙党", "きょとう", "một đảng thống nhất; tập thể đoàn kết; đảng đoàn kết")]},
    "康": {"viet": "KHANG, KHƯƠNG", "meaning_vi": "Yên. Như khang kiện [康健] yên mạnh.", "meo": "", "reading": "コウ", "vocab": [("健康", "けんこう", "khí huyết"), ("康寧", "かんやすし", "<Mỹ>")]},
    "糖": {"viet": "ĐƯỜNG", "meaning_vi": "Đường, ngày xưa dùng lúa chế ra đường tức là kẹo mạ. Đến đời Đường mới học được cách cầm mía làm đường, bên Âu châu dùng củ cải làm đường.", "meo": "", "reading": "トウ", "vocab": [("糖", "とう", "đường ."), ("乳糖", "にゅうとう", "Chất lactoza; đường sữa")]},
    "壁": {"viet": "BÍCH", "meaning_vi": "Bức vách.", "meo": "", "reading": "かべ", "vocab": [("壁", "かべ", "bức tường ."), ("内壁", "ないへき", "Tường bên trong")]},
    "燥": {"viet": "TÁO", "meaning_vi": "Khô, ráo, hanh hao.", "meo": "", "reading": "はしゃ.ぐ", "vocab": [("燥ぐ", "はしゃぐ", "làm vui vẻ; vui đùa ."), ("乾燥", "かんそう", "sự khô khan; sự nhạt nhẽo")]},
    "将": {"viet": "TƯƠNG, THƯƠNG, TƯỚNG", "meaning_vi": "Giản thể của chữ 將", "meo": "", "reading": "まさ.に はた まさ ひきい.る もって", "vocab": [("将", "はた", "người điều khiển"), ("一将", "いちしょう", "chung")]},
    "脚": {"viet": "CƯỚC", "meaning_vi": "Tục dùng như chữ cước [腳].", "meo": "", "reading": "あし", "vocab": [("脚", "あし", "cái chân"), ("三脚", "さんきゃく", "giá ba chân")]},
    "昇": {"viet": "THĂNG", "meaning_vi": "Mặt trời mới mọc.", "meo": "", "reading": "のぼ.る", "vocab": [("昇る", "のぼる", "lên cao; thăng cấp; tăng lên"), ("上昇", "じょうしょう", "sự tăng lên cao; sự tiến lên")]},
    "泥": {"viet": "NÊ, NỆ, NỄ", "meaning_vi": "Bùn.", "meo": "", "reading": "どろ", "vocab": [("泥", "どろ", "bùn"), ("泥土", "でいど", "bùn đất")]},
    "展": {"viet": "TRIỂN", "meaning_vi": "Giải, mở, bóc mở ra gọi là triển. Như phát triển [發展] mở mang rộng lớn lên, triển lãm [展覽] mở ra, bày ra cho xem.", "meo": "", "reading": "テン", "vocab": [("伸展", "しんてん", "sự mở rộng"), ("出展", "しゅってん", "vật trưng bày")]},
    "肩": {"viet": "KIÊN", "meaning_vi": "Vai.", "meo": "", "reading": "かた", "vocab": [("肩", "かた", "vai; bờ vai"), ("肩こり", "かたこり", "mỏi vai; đau vai")]},
    "態": {"viet": "THÁI", "meaning_vi": "Thái độ, thói. Như thế thái [世態] thói đời.", "meo": "", "reading": "わざ.と", "vocab": [("態", "たい", "hoàn cảnh"), ("態々", "わざわざ", "riêng để; chỉ để; cốt để; đăc biệt; làm điều gì đó một cách đặc biệt hơn là làm một cách tình cờ")]},
    "誤": {"viet": "NGỘ", "meaning_vi": "Lầm. Như thác ngộ [錯誤] lầm lẫn.", "meo": "", "reading": "あやま.る -あやま.る", "vocab": [("誤り", "あやまり", "lỗi lầm"), ("誤る", "あやまる", "lầm lỡ")]},
    "賢": {"viet": "HIỀN", "meaning_vi": "Hiền, đức hạnh tài năng hơn người gọi là hiền.", "meo": "", "reading": "かしこ.い", "vocab": [("賢い", "かしこい", "thông minh; khôn ngoan; khôn; khôn khéo"), ("賢人", "けんじん", "hiền triết")]},
    "堅": {"viet": "KIÊN", "meaning_vi": "Bền chặt.", "meo": "", "reading": "かた.い -がた.い", "vocab": [("堅い", "かたい", "cứng; vững vàng; vững chắc"), ("堅さ", "かたさ", "sự cứng; sự vững chắc; sự kiên quyết; sự khó khăn")]},
    "緊": {"viet": "KHẨN", "meaning_vi": "Khẩn cấp, gấp rút", "meo": "Quan (cái mũ) úp vào miệng để khẩn trương tuyên bố.", "reading": "きん", "vocab": [("緊急", "きんきゅう", "Khẩn cấp"), ("緊張", "きんちょう", "Căng thẳng")]},
    "狭": {"viet": "HIỆP", "meaning_vi": "Giản thể của chữ 狹", "meo": "", "reading": "せま.い せば.める せば.まる さ", "vocab": [("狭い", "せまい", "bé"), ("偏狭", "へんきょう", "hẹp hòi")]},
    "吐": {"viet": "THỔ", "meaning_vi": "Thổ ra. Vì bệnh gì mà các đồ ăn uống ở trong dạ dầy thốc ra gọi là thổ. Nhà làm thuốc có phép thổ, nghĩa là cho uống thuốc thổ hết tà độc ra cho khỏi bệnh.", "meo": "", "reading": "は.く つ.く", "vocab": [("吐く", "つく", "nói (dối); chửi"), ("吐く", "はく", "hộc")]},
    "圧": {"viet": "ÁP", "meaning_vi": "Áp lực.", "meo": "", "reading": "お.す へ.す おさ.える お.さえる", "vocab": [("圧し", "おし", "trọng lượng"), ("圧す", "おす", "hình rập nổi")]},
    "粧": {"viet": "TRANG", "meaning_vi": "trang điểm, hóa trang", "meo": "Cô Gái (女) vừa đi Sáu (六) Ngôi nhà (庄) về nên phải Trang Điểm lại.", "reading": "しょう", "vocab": [("化粧", "けしょう", "trang điểm"), ("衣装", "いしょう", "y phục, trang phục")]},
    "咲": {"viet": "TIẾU", "meaning_vi": "Tục dùng như chữ tiếu [笑].", "meo": "", "reading": "さ.く -ざき", "vocab": [("咲く", "さく", "nở"), ("咲き出す", "さきだす", "bắt đầu nở .")]},
    "値": {"viet": "TRỊ", "meaning_vi": "Cầm. Như trị kì lộ vũ [値其鷺羽] cầm thửa cánh cò.", "meo": "", "reading": "ね あたい", "vocab": [("値", "ね", "giá trị"), ("値上", "ねあげ", "sự tăng giá")]},
    "舞": {"viet": "VŨ", "meaning_vi": "Múa, cầm cái quạt hay cái nhịp múa theo âm nhạc gọi là vũ.", "meo": "", "reading": "ま.う -ま.う まい", "vocab": [("舞", "まい", "sự nhảy múa"), ("舞う", "まう", "cuộn")]},
    "我": {"viet": "NGÃ", "meaning_vi": "Ta (tiếng tự xưng mình).", "meo": "", "reading": "われ わ わ.が- わが-", "vocab": [("我", "われ", "chúng tôi ."), ("我々", "われわれ", "chúng mình")]},
    "境": {"viet": "CẢNH", "meaning_vi": "Cõi.", "meo": "", "reading": "さかい", "vocab": [("境", "さかい", "ranh giới; giới hạn; biên giới ."), ("仙境", "せんきょう", "tiên cảnh .")]},
    "鏡": {"viet": "KÍNH", "meaning_vi": "Cái gương soi, ngày xưa làm bằng đồng, bây giờ làm bằng pha lê. Nguyễn Du [阮攸] : Tha hương nhan trạng tần khai kính, Khách lộ trần ai bán độc thư [他鄉顏狀頻開鏡, 客路塵埃半讀書] (Đông lộ [東路]) Nơi quê người thường mở gương soi dung nhan, Trên đường gió bụi nơi đất khách, nửa thì giờ dùng để đọc sách. Quách Tấn dịch thơ : Đường hé quyển vàng khuây gió bụi, Trạm lau gương sáng ngắm mày râu.", "meo": "", "reading": "かがみ", "vocab": [("鏡", "かがみ", "cái gương; gương; gương soi; đèn"), ("凸鏡", "とっきょう", "Thấu kính lồi .")]},
    "看": {"viet": "KHÁN, KHAN", "meaning_vi": "Coi, xem.", "meo": "", "reading": "み.る", "vocab": [("看る", "みる", "xem; kiểm tra đánh giá; trông coi; chăm sóc"), ("准看", "じゅんかん", "sự lưu thông")]},
    "刑": {"viet": "HÌNH", "meaning_vi": "Hình phạt. Luật ngày xưa định xử tử lưu đồ trượng si [死流徒杖笞] là năm hình. Luật bây giờ chia ra hai thứ : về việc tiền của công nợ là dân sự phạm [民事犯], về việc trộm cướp đánh giết gọi là hình sự phạm [刑事犯].", "meo": "", "reading": "ケイ", "vocab": [("刑", "けい", "án; hình phạt; án phạt; bản án"), ("主刑", "しゅけい", "người phát lương")]},
    "瓶": {"viet": "BÌNH", "meaning_vi": "Như chữ bình [甁].", "meo": "", "reading": "かめ", "vocab": [("瓶", "びん", "lọ; bình"), ("土瓶", "どびん", "ấm đất")]},
    "勇": {"viet": "DŨNG", "meaning_vi": "Mạnh. Như dũng sĩ [勇士], dũng phu [勇夫].", "meo": "", "reading": "いさ.む", "vocab": [("勇", "いさむ", "tính gan dạ"), ("勇む", "いさむ", "hùng dũng; phấn khởi; hớn hở lên; hăng hái lên; quá trớn; quá đà")]},
    "両": {"viet": "LƯỠNG, LẠNG", "meaning_vi": "Tục dùng như chữ [兩].", "meo": "", "reading": "てる ふたつ", "vocab": [("両々", "りょう々", "cả hai"), ("両両", "りょうりょう", "cả hai")]},
    "違": {"viet": "VI", "meaning_vi": "Lìa. Như cửu vi [久違] ly biệt đã lâu.", "meo": "", "reading": "ちが.う ちが.い ちが.える -ちが.える たが.う たが.える", "vocab": [("違い", "ちがい", "sự khác nhau"), ("違う", "ちがう", "khác; khác nhau; không giống; trái ngược; không phù hợp")]},
    "偉": {"viet": "VĨ", "meaning_vi": "Lạ, lớn. Như tú vĩ [秀偉] tuấn tú lạ, vĩ dị [偉異] lớn lao lạ, v.v. đều là dùng để hình dung sự vật gì quí báu, hiếm có, và hình vóc cao lớn khác thường cả. Người nào có công to nghiệp lớn đều gọi là vĩ nhân [偉人].", "meo": "", "reading": "えら.い", "vocab": [("偉", "えら", "sự to lớn"), ("偉い", "えらい", "vĩ đại; tuyệt vời; giỏi")]},
    "衛": {"viet": "VỆ", "meaning_vi": "Tục dùng như chữ [衞].", "meo": "", "reading": "エイ エ", "vocab": [("侍衛", "さむらいまもる", "người hoặc nhóm người có nhiệm vụ bảo vệ một nhân vật quan trọng; vệ sĩ; đội bảo vệ"), ("衛兵", "えいへい", "vệ binh .")]},
    "充": {"viet": "SUNG", "meaning_vi": "Đầy đủ, lấp đầy", "meo": "Nhớ: Ở (儿) trong nhà (宀) thấy người (人) sung sướng vì chứa đầy của cải.", "reading": "あてる", "vocab": [("充満", "じゅうまん", "Sự tràn đầy, sự sung mãn"), ("補充", "ほじゅう", "Sự bổ sung, sự bù đắp")]},
    "銃": {"viet": "SÚNG", "meaning_vi": "Cái lỗ rìu búa để cho cán vào.", "meo": "", "reading": "つつ", "vocab": [("銃", "つつ", "Súng ."), ("銃丸", "じゅうがん", "đạn (súng trường")]},
    "統": {"viet": "THỐNG", "meaning_vi": "Mối tơ. Sự gì có manh mối có thể tìm ra được gọi là thống hệ [統系].", "meo": "", "reading": "す.べる ほび.る", "vocab": [("一統", "いっとう", "nòi giống"), ("統一", "とういつ", "sự thống nhất")]},
    "融": {"viet": "DUNG, DONG", "meaning_vi": "Sáng rực, khí lửa lan bốc lên trên trời gọi là dung. Vì thế nên ngày xưa gọi thần lửa là chúc dung thị [祝融氏].", "meo": "", "reading": "と.ける と.かす", "vocab": [("融合", "ゆうごう", "sự dung hợp ."), ("融和", "ゆうわ", "sự hài hoà")]},
    "契": {"viet": "KHẾ, TIẾT, KHIẾT, KHẤT", "meaning_vi": "Ước, làm văn tự để tin gọi là khế.", "meo": "", "reading": "ちぎ.る", "vocab": [("契り", "ちぎり", "của đợ"), ("契る", "ちぎる", "thề ước; hứa hẹn; đính hôn")]},
    "喫": {"viet": "KHIẾT", "meaning_vi": "Ăn, uống, hút", "meo": "Miệng (口) của anh hùng đại hiệp (契) lúc nào cũng phải khiết (喫) để uống rượu.", "reading": "きっ", "vocab": [("喫茶店", "きっさてん", "Quán cà phê"), ("喫煙", "きつえん", "Hút thuốc")]},
    "潔": {"viet": "KHIẾT", "meaning_vi": "Thanh khiết. Như tinh khiết [精潔] rất sạch, không lẫn lộn thứ khác.", "meo": "", "reading": "いさぎよ.い", "vocab": [("潔い", "いさぎよい", "như một người đàn ông; chơi đẹp; đầy tinh thần thể thao"), ("潔く", "いさぎよく", "như một người đàn ông")]},
    "阜": {"viet": "PHỤ", "meaning_vi": "Núi đất, đống đất, gò đất.", "meo": "", "reading": "フ フウ", "vocab": []},
    "飾": {"viet": "SỨC", "meaning_vi": "Văn sức. Vật gì đã làm, song lại trang sức thêm. Như sơn, như vẽ, như thêu, như khắc cho đẹp thêm đều gọi là sức. Như phục sức [服飾] quần áo đẹp, thủ sức [首飾] đồ trang sức trên đầu, v.v.", "meo": "", "reading": "かざ.る かざ.り", "vocab": [("飾り", "かざり", "sự giả tạo"), ("飾る", "かざる", "tô điểm")]},
    "脂": {"viet": "CHI", "meaning_vi": "Mỡ tảng, mỡ dót lại từng mảng.", "meo": "", "reading": "あぶら", "vocab": [("脂", "あぶら", "mỡ; sự khoái trá; sự thích thú"), ("松脂", "まつやに", "nhựa thông")]},
    "減": {"viet": "GIẢM", "meaning_vi": "Bớt, ít đi, giảm đi, trừ bớt đi.", "meo": "", "reading": "へ.る へ.らす", "vocab": [("減り", "へり", "sự giảm đi"), ("減る", "へる", "giảm; suy giảm; giảm bớt")]},
    "了": {"viet": "LIỄU", "meaning_vi": "Hiểu biết. Như liễu nhiên ư tâm [了然於心] lòng đã hiểu biết. Trần Nhân Tông [陳仁宗] : Niên thiếu hà tằng liễu sắc không [年少何曽了色空] (Xuân vãn [春晚]) Thời trẻ đâu hiểu được lẽ sắc không.", "meo": "", "reading": "リョウ", "vocab": [("了", "りょう", "sự kết thúc; sự hoàn thành; sự hiểu ."), ("了と", "りょうと", "sự nhận")]},
    "承": {"viet": "THỪA", "meaning_vi": "Vâng. Như bẩm thừa [稟承] bẩm vâng theo, thừa song đường chi mệnh [承雙堂之命] vâng chưng mệnh cha mẹ, v.v.", "meo": "", "reading": "うけたまわ.る う.ける ささ.げる とど.める たす.ける こ.らす つい.で すく.う", "vocab": [("承る", "うけたまわる", "tiếp nhận; chấp nhận; nghe"), ("不承", "ふしょう", "sự bất đồng quan điểm")]},
    "捜": {"viet": "SƯU", "meaning_vi": "Sưu tầm, sưu tập", "meo": "", "reading": "さが.す", "vocab": [("捜す", "さがす", "tìm kiếm"), ("捜査", "そうさ", "sự điều tra")]},
    "渡": {"viet": "ĐỘ", "meaning_vi": "Qua, từ bờ này sang bờ kia gọi là độ. Nguyễn Du [阮攸] : Vạn lý đan xa độ Hán quan [萬里單車渡漢關] (Nam Quan đạo trung [南關道中]) Trên đường muôn dặm, chiếc xe lẻ loi vượt cửa ải nhà Hán. .", "meo": "", "reading": "わた.る -わた.る わた.す", "vocab": [("渡す", "わたす", "trao"), ("渡り", "わたり", "bến phà")]},
    "盛": {"viet": "THỊNH, THÌNH", "meaning_vi": "Thịnh, đầy đủ đông đúc, chỉ thấy thêm không thấy kém đều gọi là thịnh.", "meo": "", "reading": "も.る さか.る さか.ん", "vocab": [("盛り", "さかり", "đỉnh; thời kỳ đẹp nhất; thời kỳ nở rộ; thời kỳ đỉnh cao; thời hoàng kim"), ("盛る", "さかる", "phát đạt; thịnh vượng; phát triển")]},
    "城": {"viet": "THÀNH", "meaning_vi": "Cái thành, ở trong gọi là thành [城] ở ngoài gọi là quách [郭].", "meo": "", "reading": "しろ", "vocab": [("城", "しろ", "thành; lâu đài ."), ("城主", "じょうしゅ", "chủ tòa thành; chủ lâu đài")]},
    "剤": {"viet": "TỄ", "meaning_vi": "Dịch tễ", "meo": "", "reading": "かる けず.る", "vocab": [("剤", "ざい", "thuốc ."), ("下剤", "げざい", "thuốc sổ")]},
    "武": {"viet": "VŨ, VÕ", "meaning_vi": "Võ, đối lại với văn [文]. Mạnh mẽ, chỉ chung việc làm dựa trên sức mạnh, việc quân. Như văn vũ song toàn [文武雙全] văn võ gồm tài .", "meo": "", "reading": "たけ.し", "vocab": [("武し", "ぶし", "chiến sĩ da đỏ"), ("武事", "ぶじ", "sự an toàn")]},
    "域": {"viet": "VỰC", "meaning_vi": "Bờ cõi.", "meo": "", "reading": "イキ", "vocab": [("域", "いき", "vực ."), ("区域", "くいき", "địa hạt")]},
    "裁": {"viet": "TÀI", "meaning_vi": "Cắt áo. Như tài phùng 裁縫 cắt may.", "meo": "", "reading": "た.つ さば.く", "vocab": [("裁", "さい", "quan toà"), ("裁き", "さばき", "sự xét xử")]},
    "歳": {"viet": "TUẾ", "meaning_vi": "Tuổi, năm, tuế nguyệt", "meo": "", "reading": "とし とせ よわい", "vocab": [("歳", "さい", "tuổi"), ("万歳", "ばんざい", "muôn năm")]},
    "越": {"viet": "VIỆT, HOẠT", "meaning_vi": "Qua, vượt qua. Như độ lượng tương việt [度量相越] độ lượng cùng khác nhau.", "meo": "", "reading": "こ.す -こ.す -ご.し こ.える -ご.え", "vocab": [("越し", "こし", "qua"), ("越す", "こす", "vượt qua; vượt quá")]},
    "劇": {"viet": "KỊCH", "meaning_vi": "Quá lắm. Như kịch liệt [劇烈] dữ quá, kịch đàm [劇談] bàn dữ, bệnh kịch [病劇] bệnh nặng lắm.", "meo": "", "reading": "ゲキ", "vocab": [("劇", "げき", "kịch"), ("剣劇", "けんげき", "kiếm kịch; kịch hoặc phim lấy chủ đề về kiếm thuật; kịch hoặc phim về samurai")]},
    "叫": {"viet": "KHIẾU", "meaning_vi": "Kêu, gào thét", "meo": "Miệng (口) đang kêu la (叫) vì có thứ gì kỳ lạ ở bên phải.", "reading": "さけ", "vocab": [("叫ぶ", "さけぶ", "Kêu, gào, thét"), ("絶叫", "ぜっきょう", "Sự la hét, tiếng thét")]},
    "尊": {"viet": "TÔN", "meaning_vi": "Tôn trọng. Như tôn trưởng [尊長] người tôn trưởng, tôn khách [尊客] khách quý, v.v.", "meo": "", "reading": "たっと.い とうと.い たっと.ぶ とうと.ぶ", "vocab": [("尊", "みこと", "quý"), ("尊い", "とうとい", "hiếm; quý giá")]},
    "得": {"viet": "ĐẮC", "meaning_vi": "Được. Phàm sự gì cầu mà được gọi là đắc. Nghĩ ngợi mãi mà hiểu thấu được gọi là tâm đắc [心得].", "meo": "", "reading": "え.る う.る", "vocab": [("得", "とく", "có lợi"), ("得る", "える", "đắc")]},
    "勢": {"viet": "THẾ", "meaning_vi": "Thế, chỉ về cái sức hành động. Như hỏa thế [火勢] thế lửa, thủy thế [水勢] thế nước. Tả cái hình trạng sự hành động gì cũng gọi là thế. Như trận thế [陣勢] thế trận, tư thế [姿勢] dáng bộ, v.v.", "meo": "", "reading": "いきお.い はずみ", "vocab": [("勢", "ぜい", "nghị lực"), ("勢い", "いきおい", "diễn biến (của sự kiện); xu hướng")]},
    "換": {"viet": "HOÁN", "meaning_vi": "Đổi, cải. Như cải hoán [改換] sửa đổi.", "meo": "", "reading": "か.える -か.える か.わる", "vocab": [("互換", "ごかん", "có thể thay cho nhau"), ("交換", "こうかん", "chuyển đổi")]},
    "拒": {"viet": "CỰ, CỦ", "meaning_vi": "Chống cự. Đỗ Mục [杜牧] : Sử lục quốc các ái kỳ nhân, tắc túc dĩ cự Tần [使六國各愛其人, 則足以拒秦] (A phòng cung phú [阿房宮賦]) Sáu nước nếu biết yêu thương dân mình, thì đủ sức chống lại nhà Tần.", "meo": "", "reading": "こば.む", "vocab": [("拒む", "こばむ", "từ chối; cự tuyệt; khước từ"), ("拒否", "きょひ", "sự cự tuyệt; sự phủ quyết; sự phản đối; sự phủ nhận; sự bác bỏ; cự tuyệt; phủ quyết; phản đối; phủ nhận; từ chối; bác bỏ; bác")]},
    "距": {"viet": "CỰ", "meaning_vi": "Khoảng cách, cự ly", "meo": "Chân (足) đang đi, chống cự (巨) lại khoảng cách.", "reading": "きょ", "vocab": [("距離", "きょり", "Khoảng cách, cự ly"), ("短距離", "たんきょり", "Cự ly ngắn")]},
    "覧": {"viet": "LÃM", "meaning_vi": "Tục dùng như chữ lãm [覽].", "meo": "", "reading": "み.る", "vocab": [("ご覧", "ごらん", "cái nhìn"), ("一覧", "いちらん", "cái nhìn")]},
    "蔵": {"viet": "TÀNG", "meaning_vi": "Bảo tàng, tàng trữ, tàng hình", "meo": "", "reading": "くら おさ.める かく.れる", "vocab": [("蔵", "くら", "nhà kho; sự tàng trữ; kho; cất trữ"), ("内蔵", "ないぞう", "sự lắp đặt bên trong")]},
    "臓": {"viet": "TẠNG", "meaning_vi": "Nội tạng", "meo": "", "reading": "はらわた", "vocab": [("臓", "ぞう", "nội tạng; phủ tạng"), ("五臓", "ごぞう", "ngũ tạng .")]},
    "壊": {"viet": "HOẠI", "meaning_vi": "Phá hoại", "meo": "", "reading": "こわ.す こわ.れる やぶ.る", "vocab": [("壊す", "こわす", "đánh vỡ"), ("壊乱", "かいらん", "sự hối lộ")]},
    "噴": {"viet": "PHÚN, PHÔN", "meaning_vi": "Xì ra, dùng mũi phì hơi ra.", "meo": "", "reading": "ふ.く", "vocab": [("噴く", "ふく", "thổi"), ("噴出", "ふんしゅつ", "sự phun ra (núi lửa); sự phun trào (mắc ma)")]},
    "張": {"viet": "TRƯƠNG, TRƯỚNG", "meaning_vi": "Giương. Như trương cung [張弓] giương cung.", "meo": "", "reading": "は.る -は.り -ば.り", "vocab": [("張り", "はり", "sự căng ra"), ("張る", "はる", "căng")]},
    "帳": {"viet": "TRƯỚNG", "meaning_vi": "Căng lên, dương lên. Như cung trướng [共帳] căng màn, dương màn, thông dụng như cung trướng [供帳].", "meo": "", "reading": "とばり", "vocab": [("帳", "とばり", "màn; rèm ."), ("元帳", "もとちょう", "sổ cái")]},
    "髪": {"viet": "PHÁT", "meaning_vi": "Tóc", "meo": "", "reading": "かみ", "vocab": [("髪", "かみ", "tóc"), ("一髪", "いっぱつ", "tóc")]},
    "骨": {"viet": "CỐT", "meaning_vi": "Xương, là một phần cốt yếu trong thân thể người và vật.", "meo": "", "reading": "ほね", "vocab": [("骨", "ほね", "cốt"), ("万骨", "ばんこつ", "tính không lo lắng")]},
    "塾": {"viet": "THỤC", "meaning_vi": "Cái chái nhà. Gian nhà hai bên cửa cái gọi là thục. Là chỗ để cho con em vào học, cho nên gọi chỗ chái học là gia thục [家塾]. Đời sau nhân thế mới gọi tràng học tư là tư thục [私塾], mà gọi thầy học là thục sư [塾師] vậy.", "meo": "", "reading": "ジュク", "vocab": [("塾", "じゅく", "trường tư thục ."), ("入塾", "にゅうじゅく", "sự nhập học trường tư .")]},
    "般": {"viet": "BÀN, BAN, BÁT", "meaning_vi": "Quanh co. Như bàn du [般遊] chơi quanh mãi, bàn hoàn [般桓] quấn quít không nỡ rời.", "meo": "", "reading": "ハン", "vocab": [("一般", "いっぱん", "cái chung; cái thông thường; công chúng; người dân; dân chúng"), ("今般", "こんぱん", "bây giờ")]},
    "殿": {"viet": "ĐIỆN, ĐIẾN", "meaning_vi": "Cung đền, nhà vua ở gọi là điện, chỗ thờ thần thánh cũng gọi là điện. Như cung điện [宮殿] chỗ vua ở, Phật điện [佛殿] đền thờ Phật. Ta gọi vua hay thần thánh là điện hạ [殿下] là bởi nghĩa đó.", "meo": "", "reading": "との -どの", "vocab": [("殿", "との", "cung điện; lâu đài"), ("殿", "どの", "bà; ngài.")]},
    "撃": {"viet": "KÍCH", "meaning_vi": "Như chữ [擊].", "meo": "", "reading": "う.つ", "vocab": [("撃つ", "うつ", "bắn"), ("一撃", "いちげき", "cú đánh đòn")]},
    "殺": {"viet": "SÁT, SÁI, TÁT", "meaning_vi": "Giết. Mình tự giết mình gọi là tự sát [自殺].", "meo": "", "reading": "ころ.す -ごろ.し そ.ぐ", "vocab": [("殺", "や", "giết"), ("殺し", "ころし", "tên sát nhân .")]},
    "比": {"viet": "BỈ, BÍ, BÌ, TỈ", "meaning_vi": "So sánh, lấy sự gì cùng một loài mà so sánh nhau gọi là bỉ. Về số học dùng hai số so sánh nhau để tìm số khác gọi là bỉ lệ [比例]. Về đời khoa cử gọi kỳ thi hương là đại bỉ [大比].", "meo": "", "reading": "くら.べる", "vocab": [("比", "ひ", "tỷ số"), ("比べ", "くらべ", "cuộc tranh luận")]},
    "批": {"viet": "PHÊ", "meaning_vi": "Vả, lấy tay đánh vào mặt người gọi là phê.", "meo": "", "reading": "ヒ", "vocab": [("批准", "ひじゅん", "sự phê chuẩn ."), ("批判", "ひはん", "phê phán")]},
    "需": {"viet": "NHU", "meaning_vi": "Đợi. Như tương nhu [相需] cùng đợi.", "meo": "", "reading": "ジュ", "vocab": [("内需", "ないじゅ", "Nhu cầu nội địa"), ("必需", "ひつじゅ", "cần")]},
    "端": {"viet": "ĐOAN", "meaning_vi": "Ngay thẳng.", "meo": "", "reading": "はし は はた -ばた はな", "vocab": [("端", "はし", "bờ"), ("一端", "いったん", "phần")]},
    "傾": {"viet": "KHUYNH", "meaning_vi": "Nghiêng, dốc", "meo": "Người (人) mệt mỏi (𠎚) đứng trên đất (土) xin (頁) nghỉ vì con dốc (傾).", "reading": "かたむ", "vocab": [("傾く", "かたむく", "Nghiêng, ngả"), ("傾ける", "かたむける", "Làm nghiêng, dốc")]},
    "秩": {"viet": "TRẬT", "meaning_vi": "Trật tự, thứ tự. Như trật tự [秩序] thứ hạng trên dưới trước sau.", "meo": "", "reading": "チツ", "vocab": [("秩序", "ちつじょ", "trật tự"), ("無秩序", "むちつじょ", "sự vô trật tự")]},
    "触": {"viet": "XÚC", "meaning_vi": "Giản thể của chữ 觸", "meo": "", "reading": "ふ.れる さわ.る さわ", "vocab": [("触り", "さわり", "sự sờ"), ("触る", "さわる", "chạm vào")]},
    "鹿": {"viet": "LỘC", "meaning_vi": "Con hươu. Con đực có sừng mỗi năm thay một lần, gọi là lộc nhung [鹿茸] rất bổ. Con cái không có sừng. Giống hươu sắc lông lổ đổ, nên tục gọi là mai hoa lộc [梅花鹿].", "meo": "", "reading": "しか か", "vocab": [("鹿", "しか", "hươu"), ("子鹿", "こじか", "nâu vàng")]},
    "徹": {"viet": "TRIỆT", "meaning_vi": "Suốt. Như quán triệt [貫徹] thông suốt.", "meo": "", "reading": "テツ", "vocab": [("徹す", "とおす", "trông nom"), ("一徹", "いってつ", "bướng bỉnh")]},
    "拝": {"viet": "BÁI", "meaning_vi": "bái", "meo": "", "reading": "おが.む おろが.む", "vocab": [("拝", "はい", "sự thờ cúng"), ("拝む", "おがむ", "chắp tay; mong cầu; cầu mong")]},
    "双": {"viet": "SONG", "meaning_vi": "đôi, cặp, song song", "meo": "Nhìn hai chữ 人 đứng cạnh nhau thể hiện một đôi, một cặp.", "reading": "そう", "vocab": [("双子", "ふたご", "sinh đôi"), ("双方向", "そうほうこう", "hai chiều")]},
    "隣": {"viet": "LÂN", "meaning_vi": "Lân bang, lân cận, bên cạnh", "meo": "", "reading": "とな.る となり", "vocab": [("隣", "となり", "bên cạnh"), ("隣人", "りんじん", "người láng giềng .")]},
    "災": {"viet": "TAI", "meaning_vi": "Cháy nhà.", "meo": "", "reading": "わざわ.い", "vocab": [("災い", "わざわい", "tai họa; tai ương ."), ("人災", "じんさい", "tai họa do con ngưòi tạo ra .")]},
    "離": {"viet": "LI", "meaning_vi": "Lìa tan. Lìa nhau ở gần gọi là li [離], xa gọi là biệt [別].", "meo": "", "reading": "はな.れる はな.す", "vocab": [("離す", "はなす", "bỏ; cởi"), ("不離", "ふり", "tính không thể tách rời được")]},
}


# ── N1 Kanji (1310 chữ) ─────────────────────────────────────────────────────
N1_VI: dict[str, dict] = {
    "仁": {"viet": "NHÂN", "meaning_vi": "Nhân. Nhân là cái đạo lý làm người, phải thế mới gọi là người.", "meo": "", "reading": "ジン ニ ニン", "vocab": [("仁", "じん", "lòng thương"), ("仁", "にん", "Nhân; người; thành viên")]},
    "曰": {"viet": "VIẾT", "meaning_vi": "Rằng, dùng làm lời phát ngữ.", "meo": "", "reading": "いわ.く のたま.う のたま.わく ここに", "vocab": [("曰く", "いわく", "(từ hiếm")]},
    "峠": {"viet": "(ĐÈO)", "meaning_vi": "Đỉnh núi; đèo.", "meo": "", "reading": "とうげ", "vocab": [("峠", "とうげ", "đèo"), ("峠道", "とうげみち", "đường đèo .")]},
    "洒": {"viet": "SÁI, TẨY, THỐI", "meaning_vi": "Vẩy nước rửa.", "meo": "", "reading": "すす.ぐ あら.う", "vocab": [("瀟洒", "しょうしゃ", "thanh lịch"), ("洒脱", "しゃだつ", "không theo quy ước")]},
    "酉": {"viet": "DẬU", "meaning_vi": "Chi Dậu, chi thứ mười trong mười hai chi.", "meo": "", "reading": "とり", "vocab": [("酉", "とり", "Dậu"), ("丁酉", "ていゆう", "Đinh Dậu .")]},
    "叱": {"viet": "SẤT", "meaning_vi": "Quát. Như sất trá [叱吒] quát thét. Nguyễn Du [阮攸] : Phong vũ do văn sất trá thanh [風雨猶聞叱吒聲] (Sở Bá Vương mộ [楚霸王墓]) Trong mưa gió còn nghe tiếng thét gào.", "meo": "", "reading": "しか.る", "vocab": [("叱る", "しかる", "gắt"), ("叱咤", "しった", "sự rầy la")]},
    "幌": {"viet": "HOẢNG", "meaning_vi": "Màn dũng. Đỗ Phủ [杜甫] : Hà thời ỷ hư hoảng, Song chiếu lệ ngân can [何時倚虛幌, 雙照淚痕乾] (Nguyệt dạ [月夜]) Bao giờ được tựa màn cửa trống, (Bóng trăng) chiếu hai ngấn lệ khô ?", "meo": "", "reading": "ほろ とばり", "vocab": [("幌(布の)", "ほろ（ぬのの）", "giẻ .")]},
    "汽": {"viet": "KHÍ", "meaning_vi": "Hơi nước, nước sôi bốc hơi lên gọi là khí. Như khí ki [汽機] máy hơi, khí thuyền [汽船] tàu thủy, khí xa [汽車] xe hơi, v.v.", "meo": "", "reading": "キ", "vocab": [("汽笛", "きてき", "còi xe lửa"), ("汽缶", "きかん", "người đun")]},
    "佐": {"viet": "TÁ", "meaning_vi": "Giúp. Như phụ tá [輔佐] giúp đỡ.", "meo": "", "reading": "サ", "vocab": [("佐", "さ", "sự giúp đỡ"), ("大佐", "たいさ", "đại tá .")]},
    "拓": {"viet": "THÁC, THÁP", "meaning_vi": "Nâng, lấy tay nâng đồ lên gọi là thác [拓], nay thông dụng chữ thác [托].", "meo": "", "reading": "ひら.く", "vocab": [("干拓", "かんたく", "sự khai hoang; sự khai khẩn; sự cải tạo (đất); khai hoang; khai khẩn; khai phá; khai thác"), ("拓本", "たくほん", "bản khắc (in)")]},
    "妬": {"viet": "ĐỐ", "meaning_vi": "Cũng như chữ đố [妒].", "meo": "", "reading": "ねた.む そね.む つも.る ふさ.ぐ", "vocab": [("妬く", "やく", "sự rủi ro"), ("妬み", "ねたみ", "Lòng ghen tị; sự ganh tị")]},
    "麺": {"viet": "MIẾN", "meaning_vi": "Mì sợi; bột mì", "meo": "", "reading": "むぎこ", "vocab": [("麺類", "めんるい", "bún"), ("のびる(麺が)", "のびる（めんが）", "nở .")]},
    "矢": {"viet": "THỈ", "meaning_vi": "Cái tên.", "meo": "", "reading": "や", "vocab": [("矢", "や", "mũi tên ."), ("一矢", "いっし", "tên")]},
    "撒": {"viet": "TÁT, TẢN", "meaning_vi": "Tung ra, buông ra, tòe ra. Như tát thủ [撒手] buông tay. $ Ta quen đọc là chữ tản.", "meo": "", "reading": "ま.く", "vocab": [("撒く", "まく", "rải; vẩy (nước); tưới; gieo (hạt); rắc; trải rộng"), ("撒布", "さんぷ", "số lượng những thứ được tung rắc")]},
    "絹": {"viet": "QUYÊN", "meaning_vi": "Lụa sống, lụa mộc.", "meo": "", "reading": "きぬ", "vocab": [("絹", "きぬ", "lụa; vải lụa"), ("人絹", "じんけん", "lụa nhân tạo .")]},
    "謂": {"viet": "VỊ", "meaning_vi": "Bảo, lấy lời mà bảo là vị.", "meo": "", "reading": "い.う いい おも.う いわゆる", "vocab": [("謂う", "いう", "(từ hiếm"), ("謂れ", "いわれ", "lý do")]},
    "這": {"viet": "GIÁ", "meaning_vi": "bò, trườn", "meo": "Con Trùng (虫) này mà Đến (这) gần thì phải Bò (這) trườn mà chạy.", "reading": "は", "vocab": [("這う", "はう", "bò, trườn"), ("這い出す", "はいだす", "bò ra, trườn ra")]},
    "杵": {"viet": "XỬ", "meaning_vi": "Cái chầy.", "meo": "", "reading": "きね", "vocab": [("杵", "きね", "chày ."), ("臼と杵", "うすときね", "chày cối .")]},
    "牢": {"viet": "LAO", "meaning_vi": "Cái chuồng nuôi súc vật.", "meo": "", "reading": "かた.い ひとや", "vocab": [("牢乎", "ろうこ", "hãng"), ("入牢", "にゅうろう", "Bị bỏ tù; bị vào nhà lao .")]},
    "犀": {"viet": "TÊ", "meaning_vi": "Con tê giác.", "meo": "", "reading": "サイ セイ", "vocab": [("犀利", "さいり", "sắt")]},
    "牡": {"viet": "MẪU", "meaning_vi": "Con đực, giống đực. Các chim muông thuộc về giống đực đều gọi là mẫu.", "meo": "", "reading": "おす お- おん-", "vocab": [("牡丹", "ぼたん", "mẫu đơn"), ("牡牛", "おうし", "bò cái .")]},
    "吟": {"viet": "NGÂM", "meaning_vi": "Ngâm. Đọc thơ đọc phú kéo giọng dài ra gọi là ngâm. Như ngâm nga [吟哦], ngâm vịnh [吟詠], v.v.", "meo": "", "reading": "ギン", "vocab": [("低吟", "ていぎん", "humbug"), ("吟味", "ぎんみ", "sự nếm; sự nếm thử; sự xem xét kỹ càng; sự giám định")]},
    "頷": {"viet": "HẠM", "meaning_vi": "Cằm. Chỗ cằm nở nang đầy đặn gọi là yến hạm [燕頷] cằm yến, hổ đầu yến hạm [虎頭燕頷] đầu hổ cằm yến, cổ nhân cho là cái tướng phong hầu.", "meo": "", "reading": "うなず.く あご", "vocab": [("頷く", "うなずく", "gật đầu")]},
    "貪": {"viet": "THAM", "meaning_vi": "Ăn của đút. Như tham tang uổng pháp [貪贓枉法] ăn đút làm loạn phép.", "meo": "", "reading": "むさぼ.る", "vocab": [("貪る", "むさぼる", "thèm thuồng; thèm muốn"), ("貪慾", "どんよく", "tính hám lợi")]},
    "捻": {"viet": "NIỆP, NIỆM, NẪM", "meaning_vi": "Nắn, vẽ, chữ dùng trong các từ khúc.", "meo": "", "reading": "ね.じる ねじ.る ひね.くる ひね.る", "vocab": [("捻り", "ひねり", "cái vấu"), ("捻る", "ひねる", "đánh bại")]},
    "鯰": {"viet": "NIÊM", "meaning_vi": "Dị dạng của chữ 鲇", "meo": "", "reading": "なまず", "vocab": [("鯰", "なまず", "Cá da trơn .")]},
    "嶺": {"viet": "LĨNH", "meaning_vi": "Đỉnh núi có thể thông ra đường cái được gọi là lĩnh [嶺].", "meo": "", "reading": "レイ リョウ", "vocab": [("函嶺", "かんれい", "dãy núi Hakone ."), ("高嶺", "たかね", "giọng nữ cao")]},
    "零": {"viet": "LINH", "meaning_vi": "Mưa lác đác, mưa rây.", "meo": "", "reading": "ぜろ こぼ.す こぼ.れる", "vocab": [("零", "れい", "số không ."), ("零す", "こぼす", "làm tràn; làm đổ; đánh đổ")]},
    "鈴": {"viet": "LINH", "meaning_vi": "Cái chuông rung, cái chuông nhỏ cầm tay lắc. Bạch Cư Dị [白居易] : Dạ vũ văn linh trường đoạn thanh [夜雨聞鈴腸斷聲] (Trường hận ca [長恨歌]) Đêm mưa, nghe tiếng chuông, toàn là những tiếng đoạn trường. Tản Đà dịch thơ : Đêm mưa đứt ruột, canh dài tiếng chuông.", "meo": "", "reading": "すず", "vocab": [("鈴", "すず", "chuông; cái chuông; quả chuông"), ("鈴", "れい", "số không .")]},
    "芥": {"viet": "GIỚI", "meaning_vi": "Rau cải.", "meo": "", "reading": "からし ごみ あくた", "vocab": [("塵芥", "ごみ", "vật bỏ đi"), ("芥子", "からし", "cây cải .")]},
    "謎": {"viet": "MÊ", "meaning_vi": "Câu đố.", "meo": "", "reading": "なぞ", "vocab": [("謎", "なぞ", "điều bí ẩn"), ("謎々", "なぞなぞ", "câu đố; điều bí ẩn")]},
    "襖": {"viet": "ÁO", "meaning_vi": "Áo ngắn. Như dương bì áo [羊皮襖] áo da cừu.", "meo": "", "reading": "ふすま あお", "vocab": [("襖", "ふすま", "màn kéo; tấm cửa kéo")]},
    "継": {"viet": "KẾ", "meaning_vi": "Như chữ kế [繼].", "meo": "", "reading": "つ.ぐ まま-", "vocab": [("継ぎ", "つぎ", "miếng vá"), ("継ぐ", "つぐ", "thừa kế; thừa hưởng; kế thừa .")]},
    "噛": {"viet": "GIẢO", "meaning_vi": "Cắn; nhai; gặm.", "meo": "", "reading": "か.む か.じる", "vocab": [("噛む", "かむ", "ăn khớp (bánh răng); cắn; nhai; nghiến"), ("噛る", "かじる", "nhai; cắn; ngoạm; gặm nhấm")]},
    "籾": {"viet": "(GẠO)", "meaning_vi": "hạt gạo (không vỏ).", "meo": "", "reading": "もみ", "vocab": [("籾", "もみ", "thóc")]},
    "粥": {"viet": "CHÚC, DỤC", "meaning_vi": "Cháo.", "meo": "", "reading": "かゆ かい ひさ.ぐ", "vocab": [("お粥", "おかゆ", "cháo ."), ("白粥", "はくかゆ", "cháo hoa .")]},
    "榊": {"viet": "(THẦN)", "meaning_vi": "Cây dùng để tế lễ thần đạo.", "meo": "", "reading": "さかき", "vocab": []},
    "紳": {"viet": "THÂN", "meaning_vi": "Cái dải áo to.", "meo": "", "reading": "シン", "vocab": [("紳士", "しんし", "thân sĩ; người đàn ông hào hoa phong nhã; người cao sang; người quyền quý ."), ("田紳", "でんしん", "Phú ông .")]},
    "鉢": {"viet": "BÁT", "meaning_vi": "Tiếng Phạn là bát-đa-la, là cái bát ăn của sư. Nhà chùa dùng bát xin ăn đời đời truyền để cho nhau, cho nên đời đời truyền đạo cho nhau gọi là y bát [衣鉢].", "meo": "", "reading": "ハチ ハツ", "vocab": [("鉢", "はち", "bát to"), ("丼鉢", "どんぶりはち", "cái bát")]},
    "杏": {"viet": "HẠNH", "meaning_vi": "Cây hạnh. Như ngân hạnh [銀杏] cây ngân hạnh, quả ăn được, hạt nó gọi là bạch quả [白果].", "meo": "", "reading": "あんず", "vocab": [("杏", "あんず", "cây hạnh"), ("巴旦杏", "はたんきょう", "quả mận")]},
    "呆": {"viet": "NGỐC, NGAI, BẢO", "meaning_vi": "Ngây dại. Tô Mạn Thù [蘇曼殊] : Dư ngốc lập ki bất dục sinh nhân thế [余呆立幾不欲生人世] (Đoạn hồng linh nhạn kí [斷鴻零雁記]) Tôi đứng đờ đẫn ngây dại, chẳng còn thiết chi sống ở trong nhân gian.", "meo": "", "reading": "ほけ.る ぼ.ける あき.れる おろか", "vocab": [("呆け", "ぼけ", "người ngốc nghếch; kẻ ngốc ."), ("呆ける", "ぼける", "phai (màu)")]},
    "褒": {"viet": "BAO", "meaning_vi": "Khen, ca ngợi", "meo": "Áo (衣) mặc cho người phục vụ (僕) để được khen (褒) sau khi làm việc.", "reading": "ほめる", "vocab": [("褒める", "ほめる", "Khen ngợi"), ("褒美", "ほうび", "Phần thưởng")]},
    "染": {"viet": "NHIỄM", "meaning_vi": "Nhuộm, dùng các thuốc mùi (màu) mà nhuộm các thứ đồ gọi là nhiễm. Như nhiễm liệu [染料] thuốc nhuộm.", "meo": "", "reading": "そ.める -ぞ.め -ぞめ そ.まる し.みる -じ.みる し.み -し.める", "vocab": [("染み", "しみ", "vết bẩn; vết nhơ ."), ("染め", "そめ", "sự nhuộm")]},
    "斗": {"viet": "ĐẤU, ĐẨU", "meaning_vi": "Cái đấu.", "meo": "", "reading": "ト トウ", "vocab": [("斗", "と", "to; đấu"), ("一斗", "いっと", "một đấu")]},
    "闇": {"viet": "ÁM", "meaning_vi": "Mờ tối. Như hôn ám [昏闇] tối tăm u mê.", "meo": "", "reading": "やみ くら.い", "vocab": [("闇", "やみ", "chỗ tối; nơi tối tăm; bóng tối; sự ngấm ngầm; chợ đen"), ("冥闇", "めいやみ", "tối sầm lại")]},
    "臆": {"viet": "ỨC", "meaning_vi": "Ngực. Nói bóng nghĩa là tấm lòng. Như tư ức [私臆] nỗi riêng.", "meo": "", "reading": "むね おくする", "vocab": [("臆", "おく", "rụt rè"), ("臆する", "おくする", "sự sợ")]},
    "障": {"viet": "CHƯỚNG", "meaning_vi": "Che, ngăn. Có vật gì nó làm ngăn cách gọi là chướng ngại [障礙].", "meo": "", "reading": "さわ.る", "vocab": [("障り", "さわり", "sự cản trở"), ("障る", "さわる", "trở ngại; có hại; bất lợi")]},
    "彰": {"viet": "CHƯƠNG", "meaning_vi": "Rực rỡ, lấy văn chương thêu dệt cho rực rỡ thêm gọi là chương.", "meo": "", "reading": "ショウ", "vocab": [("彰明", "あきらあきら", "bản kê khai hàng hoá chở trên tàu"), ("表彰", "ひょうしょう", "biểu chương")]},
    "詠": {"viet": "VỊNH", "meaning_vi": "Ngâm vịnh, đọc văn thơ đến chỗ có âm điệu phải kéo dài giọng đọc ra gọi là vịnh. Có khi viết là vịnh [咏].", "meo": "", "reading": "よ.む うた.う", "vocab": [("詠む", "よむ", "đếm; đọc; ngâm"), ("吟詠", "ぎんえい", "sự đọc thuộc lòng bài thơ; sự ngâm thơ .")]},
    "拉": {"viet": "Lạp", "meaning_vi": "kéo, lôi kéo, bắt cóc", "meo": "Tay (扌) đứng kéo đất (立) xuống để lắp (口) thành cái hộp.", "reading": "ラー", "vocab": [("拉致", "らち", "bắt cóc"), ("拉麺", "ラーメン", "mì ramen")]},
    "笠": {"viet": "LẠP", "meaning_vi": "Cái nón.", "meo": "", "reading": "かさ", "vocab": [("笠", "かさ", "bóng")]},
    "闊": {"viet": "KHOÁT", "meaning_vi": "Rộng rãi.", "meo": "", "reading": "ひろ.い", "vocab": [("寛闊", "かんかつ", "rộng lượng"), ("広闊", "こうかつ", "rộn lớn")]},
    "括": {"viet": "QUÁT, HOẠT", "meaning_vi": "Bao quát. Như tổng quát [總括], khái quát [概括]. Bạch Cư Dị [白居易] : Đỗ Phủ, Trần Tử Ngang, tài danh quát thiên địa [杜甫陳子昂, 才名括天地] (Sơ thụ thập di thi [初授拾遺詩]).", "meo": "", "reading": "くく.る", "vocab": [("括る", "くくる", "buộc chặt; thắt chặt; trói chặt; treo"), ("括れ", "くくれ", "sự thắt")]},
    "筈": {"viet": "", "meaning_vi": "notch of an arrow, ought, must, should be, expected", "meo": "", "reading": "はず やはず", "vocab": [("筈", "はず", "chắc chắn"), ("手筈", "てはず", "sự sắp xếp")]},
    "搾": {"viet": "TRÁ", "meaning_vi": "Bàn ép, lấy bàn ép mà ép các thứ hạt có dầu để lấy dầu dùng gọi là trá.", "meo": "", "reading": "しぼ.る", "vocab": [("搾り", "しぼり", "sự ép; vắt"), ("搾る", "しぼる", "vắt")]},
    "詐": {"viet": "TRÁ", "meaning_vi": "Giả dối.", "meo": "", "reading": "いつわ.る", "vocab": [("詐取", "さしゅ", "sự lừa gạt (tiền bạc) ."), ("詐欺", "さぎ", "sự lừa đảo")]},
    "閑": {"viet": "NHÀN", "meaning_vi": "nhàn hạ, thanh nhàn", "meo": "Trong nhà (門 - môn) có cây (木 - mộc) thì thật nhàn hạ.", "reading": "kan", "vocab": [("閑静", "かんせい", "yên tĩnh, thanh tĩnh"), ("閑散", "かんさん", "vắng vẻ, ế ẩm")]},
    "閃": {"viet": "THIỂM", "meaning_vi": "Lánh ra, nghiêng mình lánh qua gọi là thiểm.", "meo": "", "reading": "ひらめ.く ひらめ.き うかが.う", "vocab": [("閃き", "ひらめき", "sự thính"), ("閃く", "ひらめく", "lóe sáng (của chớp); lập lòe; bập bùng (của ánh lửa)")]},
    "潤": {"viet": "NHUẬN", "meaning_vi": "Nhuần, thấm, thêm.", "meo": "", "reading": "うるお.う うるお.す うる.む", "vocab": [("潤い", "うるおい", "độ ẩm; sự ướt át ."), ("潤う", "うるおう", "ẩm ướt")]},
    "欄": {"viet": "LAN", "meaning_vi": "Cùng nghĩa với chữ lan [闌] nghĩa là cái lan can. Trần Nhân Tông [陳仁宗] : Cộng ỷ lan can khán thúy vi  [共倚欄杆看翠微] (Xuân cảnh [春景]) Cùng tựa lan can ngắm khí núi xanh.", "meo": "", "reading": "てすり", "vocab": [("欄", "らん", "cột (báo)"), ("上欄", "うえらん", "sự làm mất yên tĩnh")]},
    "蘭": {"viet": "LAN", "meaning_vi": "Cây hoa lan. Có nhiều thứ, là giống hoa rất quý. Hoa lan thơm lắm, nên dầu thơm cũng gọi là lan du [蘭油]. Có thứ gọi là trạch lan [澤蘭] tức cây mần tưới trừ được mọt sách, cho nên nhà chứa sách gọi là lan tỉnh vân các [蘭省芸客], đài ngự sử gọi là lan đài [蘭臺], v.v.", "meo": "", "reading": "ラン ラ", "vocab": [("蘭国", "らんこく", "vải lanh Hà lan"), ("葉蘭", "はらん", "cây tỏi rừng .")]},
    "爛": {"viet": "LẠN", "meaning_vi": "mục nát, rực rỡ", "meo": "Gạo (米) thối (柬) vì lửa (火) gọi là mục nát (爛).", "reading": "らん", "vocab": [("爛漫", "らんまん", "rực rỡ, tươi thắm"), ("腐爛", "ふらん", "thối rữa, mục nát")]},
    "及": {"viet": "CẬP", "meaning_vi": "Kịp, đến. Từ sau mà đến gọi là cập. Như huynh chung đệ cập [兄終弟及] anh hết đến em, cập thời [及時] kịp thời, ba cập [波及] tràn tới, nghĩa bóng là sự ở nơi khác liên lụy đế mình.", "meo": "", "reading": "およ.ぶ およ.び および およ.ぼす", "vocab": [("及び", "および", "và"), ("及ぶ", "およぶ", "bằng")]},
    "扱": {"viet": "TRÁP, HẤP", "meaning_vi": "Vái chào tay sát đất.", "meo": "", "reading": "あつか.い あつか.う あつか.る こ.く", "vocab": [("扱い", "あつかい", "sự đối xử"), ("扱う", "あつかう", "điều khiển; đối phó; giải quyết")]},
    "汲": {"viet": "CẤP", "meaning_vi": "Múc nước. Nguyễn Trãi [阮薦] : Cấp giản phanh trà chẩm thạch miên [汲澗烹茶枕石眠] (Loạn hậu đáo Côn Sơn cảm tác [亂後到崑山感作]) Múc nước suối nấu trà, gối lên đá mà ngủ.", "meo": "", "reading": "く.む", "vocab": [("汲々", "きゅうきゅう", "sự siêng năng"), ("汲む", "くむ", "cùng uống rượu")]},
    "俗": {"viet": "TỤC", "meaning_vi": "Phong tục. Trên hóa kẻ dưới gọi là phong [風], dưới bắt chước trên gọi là tục [俗].", "meo": "", "reading": "ゾク", "vocab": [("俗", "ぞく", "lóng (tiếng lóng); tục; tầm thường; trần thế; thô tục"), ("俗に", "ぞくに", "thường thường")]},
    "裕": {"viet": "DỤ", "meaning_vi": "Lắm áo nhiều đồ.", "meo": "", "reading": "ユウ", "vocab": [("余裕", "よゆう", "phần dư; phần thừa ra"), ("富裕", "ふゆう", "dư dật")]},
    # "丈" → already in main DB
    "杖": {"viet": "TRƯỢNG, TRÁNG", "meaning_vi": "Cái gậy chống.", "meo": "", "reading": "つえ", "vocab": [("杖", "つえ", "cái gậy"), ("杖術", "つえじゅつ", "trước; đã nói ở trên; đã đề cập đến")]},
    "吏": {"viet": "LẠI", "meaning_vi": "quan lại, viên chức", "meo": "Một 口 (khẩu - miệng) có 一 (nhất - một vạch) nhiệm vụ - là LẠI (吏).", "reading": "リ", "vocab": [("官吏", "かんり", "quan lại"), ("吏員", "りいん", "nhân viên, viên chức")]},
    "冶": {"viet": "DÃ", "meaning_vi": "luyện kim, đúc", "meo": "Hai giọt nước (二) cùng một người (人) dùng cái cày (也) để luyện kim.", "reading": "や", "vocab": [("冶金", "やきん", "luyện kim"), ("冶具", "じぐ", "đồ gá, dụng cụ kẹp")]},
    "怠": {"viet": "ĐÃI", "meaning_vi": "Lười biếng. Như đãi nọa [怠惰] nhác nhớn.", "meo": "", "reading": "おこた.る なま.ける", "vocab": [("怠い", "だるい", "chậm chạp; uể oải; nặng nhọc"), ("怠り", "おこたり", "tính cẩu thả")]},
    "胎": {"viet": "thai", "meaning_vi": "bào thai, thai nhi", "meo": "Thịt (月) ẩn nấp bên trong đài (台) sen.", "reading": "タイ", "vocab": [("胎児", "たいじ", "thai nhi"), ("胎盤", "たいばん", "rau thai")]},
    "殆": {"viet": "ĐÃI", "meaning_vi": "Nguy. Như ngập ngập hồ đãi tai [岌岌乎殆哉] cheo leo vậy nguy thay !", "meo": "", "reading": "ほとほと ほとん.ど あやうい", "vocab": [("殆ど", "ほとんど", "hầu hết"), ("危殆", "きたい", "sự nguy hiểm")]},
    "苔": {"viet": "ĐÀI", "meaning_vi": "Rêu.", "meo": "", "reading": "こけ こけら", "vocab": [("苔", "こけ", "rêu"), ("水苔", "みずこけ", "bộ lông mao")]},
    "飴": {"viet": "DI, TỰ", "meaning_vi": "Kẹo mạch nha, kẹo mầm.", "meo": "", "reading": "あめ やしな.う", "vocab": [("飴", "あめ", "kẹo; kẹo ngậm"), ("飴玉", "あめだま", "(từ Mỹ")]},
    "唄": {"viet": "BÁI, BỐI", "meaning_vi": "Tiếng Phạn, nghĩa là chúc tụng. Bên Tây-vực có cây Bái-đa, nhà Phật dùng lá nó viết kinh gọi là bái diệp [唄葉], cũng gọi là bái-đa-la.", "meo": "", "reading": "うた うた.う", "vocab": [("唄", "うた", "tiếng hát; tiếng hót"), ("唄う", "うたう", "hát")]},
    "韻": {"viet": "VẬN", "meaning_vi": "Vần, tiếng gì đọc lên mà có vần với tiếng khác đều gọi là vận. Như công [公] với không [空] là có vần với nhau, cương [鋼] với khang [康] là có vần với nhau. Đem các chữ nó có vần với nhau chia ra từng mục gọi là vận thư [韻書] sách vần. Cuối câu thơ hay câu ca thường dùng những chữ cùng vần với nhau, luật làm thơ thì cách một câu mới dùng một vần, cho nên hai câu thơ gọi là nhất vận [一韻] một vần. Lối thơ cổ có khi mỗi câu một vần, có khi chỉ đặt luôn hai ba vần rồi đổi sang vần khác gọi là chuyển vận [轉韻] chuyển vần khác.", "meo": "", "reading": "イン", "vocab": [("韻", "いん", "vần điệu"), ("余韻", "よいん", "sự dội lại")]},
    "牲": {"viet": "SINH", "meaning_vi": "Muông sinh. Con vật nuôi gọi là súc [畜], dùng để cúng gọi là sinh [牲].", "meo": "", "reading": "セイ", "vocab": [("犠牲", "ぎせい", "sự hy sinh; phẩm chất biết hy sinh"), ("犠牲的", "ぎせいてき", "hy sinh")]},
    "醒": {"viet": "TỈNH", "meaning_vi": "Tỉnh, tỉnh cơn say.", "meo": "", "reading": "さ.ます さ.める", "vocab": [("醒める", "さめる", "lằn tàu"), ("覚醒", "かくせい", "đánh thức")]},
    "薩": {"viet": "TÁT", "meaning_vi": "Bồ tát [菩薩]. Xem chữ bồ [菩].", "meo": "", "reading": "サツ サチ", "vocab": [("菩薩", "ぼさつ", "bồ tát ."), ("地蔵菩薩", "じぞうぼさつ", "bồ tát địa tạng .")]},
    "隆": {"viet": "LONG", "meaning_vi": "Đầy ùn, đầy tù ụ, đầy đặn lại lớn lao. Vì thế nên cái gì lồi lên trội lên gọi là long khởi [隆起].", "meo": "", "reading": "リュウ", "vocab": [("隆々", "りゅうりゅう", "hưng thịnh; phồn thịnh"), ("隆昌", "りゅうしょう", "sự thịnh vượng")]},
    "蒔": {"viet": "THÌ, THI", "meaning_vi": "Thì la [蒔蘿] tức là tiểu hồi hương [小茴香] dùng để pha vào đồ ăn cho thơm.", "meo": "", "reading": "う.える まく", "vocab": [("蒔く", "まく", "gieo"), ("散蒔く", "ばらまく", "gieo rắc; phổ biến; phung phí (tiền bạc); vung (tiền)")]},
    "侍": {"viet": "THỊ", "meaning_vi": "Hầu. Như thị tọa [侍坐] ngồi hầu.", "meo": "", "reading": "さむらい はべ.る", "vocab": [("侍", "さむらい", "võ sĩ (thời cổ nhật bản); Samurai"), ("侍る", "はべる", "(thể dục")]},
    "己": {"viet": "KỈ", "meaning_vi": "Can Kỉ, can thứ sáu trong mười can.", "meo": "", "reading": "おのれ つちのと な", "vocab": [("己", "おのれ", "mày"), ("己", "つちのと", "Kỷ (hàng can) .")]},
    "紀": {"viet": "KỈ", "meaning_vi": "Gỡ sợi tơ, gỡ mối tơ rối, vì thế nên liệu lý xong công việc gọi là kinh kỉ [經紀].", "meo": "", "reading": "キ", "vocab": [("世紀", "せいき", "thế kỷ ."), ("紀元", "きげん", "kỉ nguyên")]},
    "忌": {"viet": "KỊ, KÍ", "meaning_vi": "Ghen ghét. Như đố kị [妒忌] thấy người đẹp hơn mà tức, gọi là đố [妒], thấy người giỏi hơn mà tức gọi là kị [忌].", "meo": "", "reading": "い.む い.み い.まわしい", "vocab": [("忌み", "いみ", "sự kiêng"), ("忌む", "いむ", "ghét; ghét cay ghét đắng; ghê tởm; đáng lên án")]},
    "妃": {"viet": "PHI", "meaning_vi": "Sánh đôi, cũng như chữ phối [配]. Như hậu phi [后妃] vợ cả của vua.", "meo": "", "reading": "きさき", "vocab": [("妃", "きさき", "bà chúa; bà hoàng; công chúa"), ("后妃", "こうひ", "nữ hoàng")]},
    "肢": {"viet": "CHI", "meaning_vi": "Chân tay người, chân giống thú, chân cánh giống chim đều gọi là chi.", "meo": "", "reading": "シ", "vocab": [("上肢", "じょうし", "cánh tay; chân trước của thú vật; chi trên ."), ("下肢", "かし", "chân; chi dưới")]},
    "伎": {"viet": "KỸ", "meaning_vi": "Kỹ xảo, kỹ thuật", "meo": "Người (亻) cầm cái xẻ (支) đi trêu ghẹo (又) người khác bằng kỹ năng của mình.", "reading": "ぎ", "vocab": [("演技", "えんぎ", "Diễn xuất"), ("歌舞伎", "かぶき", "Kabuki (một loại hình kịch truyền thống Nhật Bản)")]},
    "岐": {"viet": "KÌ", "meaning_vi": "Núi Kì.", "meo": "", "reading": "キ ギ", "vocab": [("分岐", "ぶんき", "sự chia nhánh; sự phân nhánh"), ("多岐", "たき", "sự lạc đề")]},
    "紋": {"viet": "VĂN", "meaning_vi": "Vân, vằn gấm vóc.", "meo": "", "reading": "モン", "vocab": [("指紋", "しもん", "dấu tay"), ("掌紋", "しょうもん", "dây đai")]},
    "蚊": {"viet": "VĂN", "meaning_vi": "Con muỗi. Có một thứ muỗi vàng đốt người hay lây bệnh sốt rét gọi là ngược môi văn [瘧媒蚊] muỗi nọc sốt rét. Nguyễn Du [阮攸] : Hư trướng tụ văn thanh [虛帳聚蚊聲] (Quế Lâm công quán [桂林公館]) Màn thưa tiếng muỗi vo ve.", "meo": "", "reading": "か", "vocab": [("蚊", "か", "con muỗi"), ("蚊屋", "かや", "bẫy muỗi .")]},
    "蘇": {"viet": "TÔ", "meaning_vi": "Tử tô [紫蘇] cây tía tô.", "meo": "", "reading": "よみがえ.る", "vocab": [("蘇り", "よみがえり", "sự làm sống lại"), ("中蘇", "ちゅうそ", "Trung Quốc và Liên Xô .")]},
    "蕾": {"viet": "LÔI", "meaning_vi": "Bội lôi [蓓蕾]. Xem chữ bội [蓓].", "meo": "", "reading": "つぼみ", "vocab": [("蕾", "つぼみ", "nụ; nụ hoa .")]},
    "鱈": {"viet": "TUYẾT", "meaning_vi": "Cá tuyết. Một giống cá sinh ở đáy bể xứ lạnh, thịt trắng như tuyết, nên gọi là tuyết ngư [鱈魚]. Trong bộ gan nó có dầu là một thứ rất bổ, ta gọi là dầu cá, ngư can du [魚肝油].", "meo": "", "reading": "たら", "vocab": [("鱈", "にしん", "cá tuyết"), ("矢鱈に", "やたらに", "hiếm khi")]},
    "雫": {"viet": "", "meaning_vi": "drop, trickle, dripping", "meo": "", "reading": "しずく", "vocab": [("雫", "しずく", "giọt .")]},
    "漏": {"viet": "LẬU", "meaning_vi": "Thấm ra, nhỏ ra, rỉ.", "meo": "", "reading": "も.る も.れる も.らす", "vocab": [("漏り", "もり", "lỗ thủng"), ("漏る", "もる", "dột")]},
    "霜": {"viet": "SƯƠNG", "meaning_vi": "Sương (vì hơi nước bốc lên gặp lạnh dót lại từng hạt nhỏ thánh thót rơi xuống gọi là sương. Nguyễn Du [阮攸] : Thu mãn phong lâm sương diệp hồng [秋滿楓林霜葉紅] (Từ Châu đạo trung [徐州道中]) Thu ngập rừng phong, sương nhuộm đỏ lá.", "meo": "", "reading": "しも", "vocab": [("霜", "しも", "sương"), ("初霜", "はつしも", "màn sương đầu tiên trong năm")]},
    "搭": {"viet": "ĐÁP", "meaning_vi": "Phụ vào, đáp đi. Như đáp xa [搭車] đạp xe đi, đáp thuyền [搭船] đáp thuyền đi, v.v.", "meo": "", "reading": "トウ", "vocab": [("搭乗", "とうじょう", "việc lên máy bay"), ("搭載", "とうさい", "sự trang bị; sự lắp đặt kèm theo .")]},
    "塔": {"viet": "THÁP", "meaning_vi": "Cái tháp. Nguyên âm tiếng Phạn là tháp bà [塔婆] hay tốt đổ ba [窣睹波]. Nguyễn Trãi [阮廌] : Tháp ảnh trâm thanh ngọc  [塔影簪青玉] (Dục Thúy sơn [浴翠山]) Bóng tháp cài trâm ngọc xanh.", "meo": "", "reading": "トウ", "vocab": [("塔", "とう", "đài"), ("仏塔", "ぶっとう", "Chùa .")]},
    "孟": {"viet": "MẠNH, MÃNG", "meaning_vi": "Lớn, con trai trưởng dòng đích gọi là bá [伯], con trai trưởng dòng thứ gọi là mạnh [孟].", "meo": "", "reading": "かしら", "vocab": []},
    "猛": {"viet": "MÃNH", "meaning_vi": "Mạnh. Như mãnh tướng [猛將] tướng mạnh, mãnh thú [猛獸] thú mạnh, v.v.", "meo": "", "reading": "モウ", "vocab": [("猛し", "もうし", "chiến sĩ da đỏ"), ("兇猛", "きょうもう", "hung dữ")]},
    "溢": {"viet": "DẬT", "meaning_vi": "Đầy tràn. Hiếu Kinh có câu mãn nhi bất dật [滿而不溢] ý nói giàu mà không kiêu xa.", "meo": "", "reading": "こぼ.れる あふ.れる み.ちる", "vocab": [("溢れ", "あふれ", "sự tràn ra"), ("充溢", "じゅういつ", "sự tràn ra")]},
    "薪": {"viet": "TÂN", "meaning_vi": "Củi. Như mễ châu tân quế [米珠薪桂] gạo châu củi quế.", "meo": "", "reading": "たきぎ まき", "vocab": [("薪", "たきぎ", "củi"), ("薪割", "たきぎわり", "cái rìu nhỏ")]},
    "忠": {"viet": "TRUNG", "meaning_vi": "Thực, dốc lòng, hết bổn phận mình là trung [忠].", "meo": "", "reading": "チュウ", "vocab": [("不忠", "ふちゅう", "bất trung ."), ("忠信", "ちゅうしん", "lòng trung thành")]},
    "串": {"viet": "XUYẾN, QUÁN", "meaning_vi": "Suốt, một quan tiền gọi là nhất xuyến [一串], cái giấy biên thu tiền gọi là xuyến phiếu [串票].", "meo": "", "reading": "くし つらぬ.く", "vocab": [("串刺し", "くしざし", "cái xiên")]},
    "伴": {"viet": "BẠN", "meaning_vi": "Bạn. Như đồng bạn [同伴] người cùng ăn với mình.", "meo": "", "reading": "ともな.う", "vocab": [("伴", "とも", "bạn"), ("伴う", "ともなう", "dìu dắt")]},
    "畔": {"viet": "BẠN", "meaning_vi": "Bờ ruộng.", "meo": "", "reading": "あぜ くろ ほとり", "vocab": [("河畔", "かはん", "ven sông"), ("渚畔", "なぎさほとり", "bờ (biển")]},
    "絆": {"viet": "BÁN", "meaning_vi": "Cùm ngựa.", "meo": "", "reading": "きずな ほだ.す つな.ぐ", "vocab": [("絆", "きずな", "gánh nặng"), ("羈絆", "きはん", "dây đai")]},
    "倹": {"viet": "KIỆM", "meaning_vi": "Tiết kiệm", "meo": "", "reading": "つま.しい つづまやか", "vocab": [("倹", "けん", "kinh tế; tiết kiệm"), ("勤倹", "きんけん", "sự cần kiệm; cần kiệm; tiết kiệm")]},
    "剣": {"viet": "KIẾM", "meaning_vi": "Thanh kiếm", "meo": "", "reading": "つるぎ", "vocab": [("剣", "つるぎ", "kiếm ."), ("刀剣", "とうけん", "đao")]},
    "鹸": {"viet": "", "meaning_vi": "saltiness", "meo": "", "reading": "あ.く", "vocab": [("石鹸", "せっけん", "xà phòng ."), ("粉石鹸", "こなせっけん", "xà phòng bột .")]},
    "婆": {"viet": "BÀ", "meaning_vi": "Bà, đàn bà già gọi là bà. Tục gọi mẹ chồng là bà.", "meo": "", "reading": "ばば ばあ", "vocab": [("婆", "ばば", "mụ phù thuỷ"), ("婆あ", "ばばあ", "mụ phù thuỷ")]},
    "披": {"viet": "PHI, BIA", "meaning_vi": "Vạch ra, xé ra. Như phi vân kiến nhật [披雲見日] vạch mây thấy mặt trời. Tô Thức [蘇軾] : Phi mông nhung [披蒙茸] (Hậu Xích Bích phú [後赤壁賦]) Rẽ đám cỏ rậm rạp.", "meo": "", "reading": "ヒ", "vocab": [("披歴", "ひれき", "sự bộc lộ ."), ("披瀝", "ひれき", "trạng thái")]},
    "志": {"viet": "CHÍ", "meaning_vi": "Nơi để tâm vào đấy gọi là chí. Như hữu chí cánh thành [有志竟成] có chí tất nên. Người có khí tiết gọi là chí sĩ [志士] nghĩa là tâm có chủ trương, không có a dua theo đời vậy.", "meo": "", "reading": "シリング こころざ.す こころざし", "vocab": [("志", "こころざし", "lòng biết ơn"), ("志す", "こころざす", "ước muốn; ý muốn")]},
    "苺": {"viet": "MÔI", "meaning_vi": "Như chữ môi [莓].", "meo": "", "reading": "いちご", "vocab": [("苺", "いちご", "dâu tây; quả dâu tây; cây dâu tây"), ("木苺", "きいちご", "cây dâu rừng; dâu rừng .")]},
    "悔": {"viet": "HỐI, HỔI", "meaning_vi": "Hối hận, biết lỗi mà nghĩ cách đổi gọi là hối. Vương An Thạch [王安石] : Dư diệc hối kì tùy chi, nhi bất đắc cực hồ du chi lạc dã [予亦悔其隨之, 而不得極乎遊之樂也] (Du Bao Thiền Sơn kí [遊褒禪山記]) Tôi cũng ân hận rằng đã theo họ, không được thỏa hết cái thú vui du lãm.", "meo": "", "reading": "く.いる く.やむ くや.しい", "vocab": [("悔い", "くい", "sự ăn năn; sự hối hận; sự hối lỗi; sự sám hối; sự ân hận; ăn năn; hối hận; hối lỗi; sám hối; ân hận"), ("悔み", "くやみ", "lời chia buồn")]},
    "侮": {"viet": "VŨ", "meaning_vi": "Khinh nhờn. Như khi vũ [欺侮] lừa gạt hà hiếp. Nguyễn Du [阮攸] : Nại hà vũ quả nhi khi cô [奈何侮寡而欺孤] (Cựu Hứa đô [舊許都]) Sao lại đi lừa vợ góa dối con côi người ta (nói về Tào Tháo [曹操]) ?", "meo": "", "reading": "あなど.る あなず.る", "vocab": [("侮", "あなど", "xem thường"), ("侮り", "あなどり", "sự coi khinh")]},
    "敏": {"viet": "MẪN", "meaning_vi": "Nhanh nhẹn. Như mẫn tiệp [敏捷] nhanh nhẹn.", "meo": "", "reading": "さとい", "vocab": [("不敏", "ふびん", "sự không có khả năng"), ("俊敏", "しゅんびん", "nhanh nhạy; thông minh sắc sảo")]},
    "繁": {"viet": "PHỒN, BÀN", "meaning_vi": "Nhiều. Như phồn thịnh [繁盛] nhiều nhõi đông đúc, phồn diễn [繁衍] nhung nhúc, đầy đàn, đầy lũ.", "meo": "", "reading": "しげ.る しげ.く", "vocab": [("繁忙", "はんぼう", "bận rộn"), ("繁昌", "はんじょう", "Sự thịnh vượng; sự hưng thịnh .")]},
    "芯": {"viet": "TÂM", "meaning_vi": "Bấc đèn. Ruột một thứ cỏ dùng để thắp đèn gọi là đăng tâm [燈芯].", "meo": "", "reading": "シン", "vocab": [("芯", "しん", "bấc"), ("芯地", "しんじ", "sự đệm")]},
    "秘": {"viet": "BÍ", "meaning_vi": "Tục dùng như chữ bí [祕].", "meo": "", "reading": "ひ.める ひそ.か かく.す", "vocab": [("秘か", "ひそか", "Bí mật; riêng tư; lén lút ."), ("秘中", "ひちゅう", "trong vòng bí mật .")]},
    "泌": {"viet": "BÍ", "meaning_vi": "Sông Bí, thuộc tỉnh Hà Nam [河南].", "meo": "", "reading": "ヒツ ヒ", "vocab": [("分泌", "ぶんぴ", "sự cất giấu; sự giấu giếm; sự bưng bít"), ("内分泌", "ないぶんぴ", "sự bí mật nội bộ")]},
    "密": {"viet": "MẬT", "meaning_vi": "Rậm rạp, liền kín. Như mật mật tằng tằng [密密層層] chập chồng liền kín, mật như thù võng [密如蛛網] dày đặc như mạng nhện.", "meo": "", "reading": "ひそ.か", "vocab": [("密か", "ひそか", "sự thầm kín; sự bí mật"), ("密事", "みつじ", "kín đáo")]},
    "蜜": {"viet": "MẬT", "meaning_vi": "Mật ong.", "meo": "", "reading": "ミツ ビツ", "vocab": [("蜜", "みつ", "(thần thoại"), ("蜜柑", "みかん", "quýt; quả quýt .")]},
    "芳": {"viet": "PHƯƠNG", "meaning_vi": "Cỏ thơm. Như phương thảo [芳草] cỏ thơm.", "meo": "", "reading": "かんば.しい", "vocab": [("余芳", "よかおる", "sự dự đoán trước; sự dự báo trước"), ("芳名", "ほうめい", "danh thơm; danh tiếng tốt .")]},
    "妨": {"viet": "PHƯƠNG, PHƯỚNG", "meaning_vi": "Hại, ngại. Nguyễn Du [阮攸] : Bất phương chung nhật đối phù âu [不妨終日對浮鷗] (Hoàng Hà trở lạo [黄河阻潦]) Không ngại gì, cả ngày đối mặt với đám chim âu. $ Có khi đọc là phướng.", "meo": "", "reading": "さまた.げる", "vocab": [("妨げ", "さまたげ", "sự làm tắc nghẽn"), ("妨害", "ぼうがい", "sự phương hại; sự cản trở")]},
    "紡": {"viet": "PHƯỞNG", "meaning_vi": "Các thứ dệt bằng tơ đông đặc mềm nhũn tục gọi là phưởng trù [紡綢].", "meo": "", "reading": "つむ.ぐ", "vocab": [("紡ぐ", "つむぐ", "kéo sợi"), ("混紡", "こんぼう", "chỉ hỗn hợp .")]},
    "房": {"viet": "PHÒNG, BÀNG", "meaning_vi": "Cái buồng.", "meo": "", "reading": "ふさ", "vocab": [("房", "ふさ", "búi; chùm"), ("乳房", "にゅうぼう", "Vú .")]},
    "倣": {"viet": "PHỎNG", "meaning_vi": "phỏng theo, bắt chước", "meo": "Người (亻) PHƯƠNG (方) pháp bắt chước là tốt nhất.", "reading": "narau", "vocab": [("模倣", "mohō", "mô phỏng"), ("倣う", "narau", "bắt chước, phỏng theo")]},
    "傍": {"viet": "BÀNG, BẠNG", "meaning_vi": "Bên. Cũng như chữ bàng [旁].", "meo": "", "reading": "かたわ.ら わき おか- はた そば", "vocab": [("傍", "はた", "gần"), ("傍ら", "かたわら", "bên cạnh; gần sát")]},
    "謗": {"viet": "BÁNG", "meaning_vi": "Chê bai, báng bổ, thấy người làm việc trái mà mọi người cùng xúm lại chê bai mai mỉa gọi là báng. Nguyễn Trãi [阮廌] : Chúng báng cô trung tuyệt khả liên [眾謗孤忠絕可憐] (Oan thán [冤嘆]) Bao kẻ gièm pha, người trung cô lập, thực đáng thương.", "meo": "", "reading": "そし.る", "vocab": [("謗り", "そしり", "sự vu cáo"), ("謗る", "そしる", "sự vu cáo")]},
    "傲": {"viet": "NGẠO", "meaning_vi": "kiêu ngạo, ngạo mạn", "meo": "Người (亻) hay đội mũ (白) mỗi ngày (每) thì coi chừng bị ngạo mạn, kiêu căng.", "reading": "おご", "vocab": [("傲慢", "ごうまん", "kiêu ngạo, ngạo mạn"), ("傲然", "ごうぜん", "kiêu căng, ngạo nghễ")]},
    "腸": {"viet": "TRÀNG, TRƯỜNG", "meaning_vi": "Ruột. Phần nhỏ liền với dạ dầy gọi là tiểu tràng [小腸] ruột non, phần to liền với lỗ đít gọi là đại tràng [大腸] ruột già. Còn đọc là trường.", "meo": "", "reading": "はらわた わた", "vocab": [("腸", "ちょう", "ruột"), ("腸", "わた", "nội tạng của loài cá .")]},
    "揚": {"viet": "DƯƠNG", "meaning_vi": "Giơ lên, bốc lên. Như thủy chi dương ba [水之揚波] nước chưng gợn sóng, phong chi dương trần [風之揚塵] gió chưng bốc bụi lên, v.v. Nguyễn Dữ [阮嶼] : Ngã tào du thử cận bát vạn niên, nam minh dĩ tam dương trần hĩ [我曹遊此僅八萬年, 南溟已三揚塵矣] (Từ Thức tiên hôn lục [徐式僊婚綠]) Chúng tôi chơi ở chốn này mới tám vạn năm, mà bể Nam đã ba lần tung bụi.", "meo": "", "reading": "あ.げる -あ.げ あ.がる", "vocab": [("揚句", "あげく", "cuối cùng"), ("大揚", "だいよう", "tính rộng rãi")]},
    "瘍": {"viet": "DƯƠNG", "meaning_vi": "Phàm các bệnh nhọt sảy đều gọi là dương, nên thầy thuốc ngoại khoa gọi là dương y [瘍醫].", "meo": "", "reading": "かさ", "vocab": [("潰瘍", "かいよう", "Loét; chỗ loét"), ("腫瘍", "しゅよう", "bệnh sưng lên .")]},
    "暢": {"viet": "SƯỚNG", "meaning_vi": "Sướng. Như thông sướng [通暢] thư sướng, thông suốt, sướng khoái [暢快] sướng thích.", "meo": "", "reading": "のび.る", "vocab": [("伸暢", "しんとおる", "sự mở rộng"), ("暢気", "のんき", "sự vô lo")]},
    "幻": {"viet": "HUYỄN, ẢO", "meaning_vi": "Dối giả, làm giả mê hoặc người.", "meo": "", "reading": "まぼろし", "vocab": [("幻", "まぼろし", "ảo tưởng; ảo vọng; ảo ảnh; ảo mộng ."), ("幻像", "げんぞう", "ảo tưởng; giấc mơ; giấc mộng")]},
    "拗": {"viet": "ẢO, ÁO, HÚC", "meaning_vi": "Bẻ.", "meo": "", "reading": "ねじ.れる こじ.れる す.ねる ねじ.ける", "vocab": [("執拗", "しつよう", "tính bướng bỉnh"), ("拗ける", "ねじける", "đường cong")]},
    "紛": {"viet": "PHÂN", "meaning_vi": "Rối rít.", "meo": "", "reading": "まぎ.れる -まぎ.れ まぎ.らす まぎ.らわす まぎ.らわしい", "vocab": [("紛れ", "まぐれ", "sự may mắn; cơ may"), ("紛乱", "ふんらん", "sự lộn xôn")]},
    "雰": {"viet": "PHÂN", "meaning_vi": "Khí sương mù.", "meo": "", "reading": "フン", "vocab": [("雰囲気", "ふんいき", "bầu không khí"), ("犯罪を雰囲気", "はんざいをふんいき", "sát khí .")]},
    "盆": {"viet": "BỒN", "meaning_vi": "Cái bồn, cái chậu sành. Trang Tử cổ bồn ca [莊子鼓盆歌] Trang Tử đánh vào cái bồn mà hát.", "meo": "", "reading": "ボン", "vocab": [("盆", "ぼん", "mâm; khay ."), ("お盆", "おぼん", "lễ Obon")]},
    "頒": {"viet": "BAN", "meaning_vi": "Phân phát, ban hành", "meo": "Chữ 頭 (đầu) bị chia 八 (tám) lần cho 貝 (vỏ sò/tiền bối)", "reading": "はん", "vocab": [("頒布", "はんぷ", "Phân bố, phát hành"), ("頒価", "はんか", "Giá bán, giá phân phối")]},
    "寡": {"viet": "QUẢ", "meaning_vi": "góa, ít", "meo": "Ít người (人) trùm MÁI NHÀ (宀) ngồi chia CỦ (分) vì nghèo nên phải ở vậy.", "reading": "か", "vocab": [("寡婦", "かふ", "góa phụ"), ("寡占", "かせん", "độc chiếm, thiểu số chiếm lĩnh")]},
    "采": {"viet": "THẢI, THÁI", "meaning_vi": "Hái, ngắt.", "meo": "", "reading": "と.る いろどり", "vocab": [("喝采", "かっさい", "sự hoan hô nhiệt liệt"), ("納采", "のうさい", "Quà tặng hứa hôn .")]},
    "免": {"viet": "MIỄN, VẤN", "meaning_vi": "Bỏ. Như miễn quan [免冠] trật mũ.", "meo": "", "reading": "まぬか.れる まぬが.れる", "vocab": [("免", "めん", "sự giải tán"), ("ご免", "ごめん", "Xin hãy tha thứ!; Xin lỗi!")]},
    "逸": {"viet": "DẬT", "meaning_vi": "Lầm lỗi. Như dâm dật [淫逸] dâm dục quá độ.", "meo": "", "reading": "そ.れる そ.らす はぐ.れる", "vocab": [("逸事", "いつじ", "chuyện vặt"), ("俊逸", "しゅんいつ", "sự trội hơn")]},
    "挽": {"viet": "VÃN", "meaning_vi": "Kéo lại. Như vãn hồi [挽回] xoay lại, vãn lưu [挽留] kéo giữ lại, v.v.", "meo": "", "reading": "ひ.く", "vocab": [("挽く", "ひく", "xay"), ("挽回", "ばんかい", "Sự khôi phục; sự phục hồi; sự vãn hồi; sự cứu vãn tình thế .")]},
    "娩": {"viet": "VÃN, MIỄN", "meaning_vi": "Uyển vãn [婉娩] thùy mị, tả cái nét con gái nhu thuận.", "meo": "", "reading": "ベン", "vocab": [("分娩", "ぶんべん", "sự phân phát ; sự phân phối"), ("分娩室", "ぶんべんしつ", "phòng đẻ .")]},
    "喰": {"viet": "", "meaning_vi": "eat, drink, receive (a blow), (kokuji)", "meo": "", "reading": "く.う く.らう", "vocab": [("喰う", "くう", "ăn; (từ Mỹ"), ("喰らう", "くらう", "ăn; (từ Mỹ")]},
    "狼": {"viet": "LANG", "meaning_vi": "Con chó sói. Tính tàn ác như hổ, cho nên gọi các kẻ tàn bạo là lang hổ [狼虎].", "meo": "", "reading": "おおかみ", "vocab": [("狼", "おおかみ", "chó sói; sói"), ("狼火", "ろうか", "đèn hiệu")]},
    "郎": {"viet": "LANG", "meaning_vi": "Chức quan. Về đời nhà Tần [秦], nhà Hán [漢] thì các quan về hạng lang đều là sung vào quan túc vệ. Về đời sau mới dùng để gọi các quan ngoài, như thượng thư lang [尚書郎], thị lang [侍郎], v.v. Ở bên ta thì các quan cai trị thổ mán đều gọi đều gọi là quan lang.", "meo": "", "reading": "おとこ", "vocab": [("下郎", "げろう", "người hầu"), ("郎党", "ろうどう", "lão bộc; quản gia; người hầu cận; người tùy tùng; đầy tớ .")]},
    "朗": {"viet": "LÃNG", "meaning_vi": "Sáng. Như thiên sắc thanh lãng [天色清朗] màu trời trong sáng.", "meo": "", "reading": "ほが.らか あき.らか", "vocab": [("朗々", "ろうろう", "trong"), ("朗吟", "ろうぎん", "sự kể lại")]},
    "浪": {"viet": "LÃNG, LANG", "meaning_vi": "Sóng.", "meo": "", "reading": "ロウ", "vocab": [("浪々", "ろうろう", "sự đi lang thang"), ("浪人", "ろうにん", "lãng tử; kẻ vô công rồi nghề; kẻ lang thang")]},
    "痕": {"viet": "NGÂN", "meaning_vi": "Sẹo, vết. Phàm vật gì có dấu vết đều gọi là ngân. Như mặc ngân [墨痕] vết mực. Đỗ Phủ [杜甫] : Hà thời ỷ hư hoảng, Song chiếu lệ ngân can [何時倚虛幌, 雙照淚痕乾] (Nguyệt dạ [月夜]) Bao giờ được tựa màn cửa trống, (Bóng trăng) chiếu hai ngấn lệ khô ?", "meo": "", "reading": "あと", "vocab": [("痕", "あと", "vết ."), ("傷痕", "きずあと", "vết sẹo; vết thẹo; sẹo; thẹo")]},
    "恨": {"viet": "HẬN", "meaning_vi": "Oán giận. Sự gì đã mất hy vọng thực gọi là hận. Như hận sự [恨事] việc đáng giận, di hận [遺恨] để sự giận lại, ẩm hận [飲恨] nuốt hận, v.v. Nguyễn Trãi [阮廌] : Anh hùng di hận kỷ thiên niên [英雄遺恨幾千年] (Quan hải [關海]) Anh hùng để lại mối hận đến mấy nghìn năm.", "meo": "", "reading": "うら.む うら.めしい", "vocab": [("恨み", "うらみ", "mối hận; sự căm ghét"), ("恨む", "うらむ", "hận; căm ghét; khó chịu; căm tức")]},
    "腿": {"viet": "THỐI", "meaning_vi": "Đùi. Đùi vế gọi là đại thối [大腿], bắp chân gọi là tiểu thối [小腿]. Nguyên viết là thối [骽].", "meo": "", "reading": "もも", "vocab": [("腿", "もも", "bắp đùi"), ("上腿", "じょうたい", "bắp đùi")]},
    "爵": {"viet": "TƯỚC", "meaning_vi": "Cái chén rót rượu.", "meo": "", "reading": "シャク", "vocab": [("伯爵", "はくしゃく", "bá tước ."), ("爵位", "しゃくい", "tước vị; chức tước .")]},
    "郷": {"viet": "HƯƠNG", "meaning_vi": "quê hương.", "meo": "", "reading": "さと", "vocab": [("仙郷", "せんきょう", "tiên giới"), ("郷俗", "さとぞく", "côn đồ; kẻ hung ác")]},
    "櫛": {"viet": "TRẤT", "meaning_vi": "Cái lược.", "meo": "", "reading": "くし くしけず.る", "vocab": [("櫛", "くし", "lược chải đầu; lược"), ("櫛風", "しっぷう", "cơn gió mạnh .")]},
    "祥": {"viet": "TƯỜNG", "meaning_vi": "Điềm, điềm tốt gọi là tường [祥], điềm xấu gọi là bất tường [不祥].", "meo": "", "reading": "さいわ.い きざ.し よ.い つまび.らか", "vocab": [("不祥", "ふしょう", "ô nhục"), ("健祥", "けんさち", "tinh thần")]},
    "翔": {"viet": "TƯỜNG", "meaning_vi": "Liệng quanh. Nguyễn Du [阮攸] : Đương thế hà bất nam du tường [當世何不南遊翔] (Kỳ lân mộ [騏麟墓]) Thời ấy sao không bay lượn sang Nam chơi ?", "meo": "", "reading": "かけ.る と.ぶ", "vocab": [("翔る", "かける", "sự bay vút lên"), ("翔ける", "かける", "sự bay vút lên")]},
    "姜": {"viet": "KHƯƠNG", "meaning_vi": "Họ vua Khương [姜]. Vua Thần Nông [神農] ở bên sông Khương, nhân lấy tên sông làm họ.", "meo": "", "reading": "こう", "vocab": [("生姜", "しょうが", "gừng .")]},
    "窯": {"viet": "DIÊU", "meaning_vi": "Cái lò, cái lò nung vôi nung ngói, nung các đồ sứ, vì thế nên các đồ sành đồ sứ gọi là diêu.", "meo": "", "reading": "かま", "vocab": [("窯", "かま", "lò; lò nung"), ("窯元", "かまもと", "đồ gốm")]},
    "云": {"viet": "VÂN", "meaning_vi": "Rằng. Như ngữ vân [語云] nhời quê nói rằng.", "meo": "", "reading": "い.う ここに", "vocab": [("云々", "うんぬん", "lời bình luận"), ("云う", "いう", "(từ hiếm")]},
    "陰": {"viet": "ÂM", "meaning_vi": "Số âm, phần âm, trái lại với chữ dương [陽]. Phàm sự vật gì có thể đối đãi lại, người xưa thường dùng hai chữ âm dương [陰陽] mà chia ra. Như trời đất, mặt trời mặt trăng, rét nóng, ngày đêm, trai gái, trong ngoài, cứng mềm, động tĩnh, v.v. đều chia phần này là dương, phần kia là âm. Vì các phần đó nó cùng thêm bớt thay đổi nhau, cho nên lại dùng để xem tốt xấu nữa. Từ đời nhà Hán [漢] trở lên thì những nhà xem thuật số đều gọi là âm dương gia [陰陽家].", "meo": "", "reading": "かげ かげ.る", "vocab": [("陰", "いん", "âm (dương)"), ("陰", "かげ", "bóng tối; sự tối tăm; u ám")]},
    "蔭": {"viet": "ẤM", "meaning_vi": "Bóng cây, bóng rợp.", "meo": "", "reading": "かげ", "vocab": [("蔭", "かげ", "bóng"), ("お蔭", "おかげ", "sự giúp đỡ; sự ủng hộ; nhờ vào")]},
    "曇": {"viet": "ĐÀM", "meaning_vi": "Mây chùm (mây bủa).", "meo": "", "reading": "くも.る", "vocab": [("曇", "くもり", "trời đầy mây; u ám"), ("曇り", "くもり", "mờ; không rõ; nhiều mây")]},
    "魂": {"viet": "HỒN", "meaning_vi": "Phần hồn, là cái làm chúa tể cả phần tinh thần. Người ta lúc sống thì hồn phách cùng quấn với nhau, đến lúc chết thì hồn phách lìa nhau. Vì thế mới bảo thần với quỷ đều là hồn hóa ra cả, vì nó là một vật rất thiêng, thiêng hơn cả muôn vật, cho nên lại gọi là linh hồn [靈魂].", "meo": "", "reading": "たましい たま", "vocab": [("魂", "こん", "Linh hồn; tinh thần"), ("魂", "たましい", "linh hồn .")]},
    "釜": {"viet": "PHỦ", "meaning_vi": "Cái nồi, cái chảo, cái chõ.", "meo": "", "reading": "かま", "vocab": [("釜", "かま", "ấm đun nước; ấm tích; nồi đun; nồi nấu; lò đun; lò sấy; lò nung; lò (nung vôi"), ("お釜", "おかま", "người đồng tính luyến ái nam; đồng tính; pêđê; ái nam ái nữ")]},
    "斧": {"viet": "PHỦ", "meaning_vi": "Cái búa.", "meo": "", "reading": "おの", "vocab": [("斧", "おの", "cái rìu"), ("手斧", "ちょうな", "rìu lưỡi vòm; rìu lưỡi")]},
    "郊": {"viet": "GIAO", "meaning_vi": "Chỗ cách xa nước một trăm dặm. Nay thường gọi ngoài thành là cận giao [近郊] cõi gần thành.", "meo": "", "reading": "コウ", "vocab": [("北郊", "ほっこう", "về hướng bắc"), ("南郊", "なんこう", "Vùng ngoại ô ở phía Nam .")]},
    "絞": {"viet": "GIẢO, HÀO", "meaning_vi": "Vắt, thắt chặt. Như giảo thủ cân [絞毛巾] vắt khăn tay.", "meo": "", "reading": "しぼ.る し.める し.まる", "vocab": [("絞る", "しぼる", "vắt (quả) ."), ("お絞り", "おしぼり", "khăn bông ướt để lau tay ở bàn ăn trong nhà hàng")]},
    "鮫": {"viet": "GIAO", "meaning_vi": "Cá giao, vây nó ăn rất ngon. Có khi gọi là sa ngư [沙魚].", "meo": "", "reading": "さめ みずち", "vocab": [("鮫", "さめ", "cá đao"), ("鮫皮", "さめがわ", "da cá mập")]},
    "咬": {"viet": "GIẢO", "meaning_vi": "Cắn vào xương. Như giảo nha [咬牙] nghiến răng. Nguyên là chữ giảo [齩].", "meo": "", "reading": "か.む", "vocab": [("咬み", "かみ", "vết cắn; cắn"), ("咬む", "かむ", "cắn; nhai; gặm")]},
    "懸": {"viet": "HUYỀN", "meaning_vi": "Treo, treo thằng lẵng giữa khoảng không gọi là huyền. Như huyền nhai [懸崖] sườn núi dốc đứng (như treo lên).", "meo": "", "reading": "か.ける か.かる", "vocab": [("懸吊", "かかつ", "sự treo"), ("懸命", "けんめい", "sự ham")]},
    "迫": {"viet": "BÁCH", "meaning_vi": "Gần sát. Thời gian hay địa thế kề sát tận nơi rồi không còn một khe nào nữa gọi là bách. Vì thế nên sự cần kíp lắm gọi là quẫn bách [窘迫].", "meo": "", "reading": "せま.る", "vocab": [("迫る", "せまる", "cưỡng bức; giục; thúc giục"), ("切迫", "せっぱく", "sự sắp xảy ra; sự đang đe dọa; sự khẩn cấp; sự cấp bách .")]},
    "伯": {"viet": "BÁ", "meaning_vi": "Bác, anh bố gọi là bá phụ [伯父]. Đàn bà gọi anh chồng là bá.", "meo": "", "reading": "ハク", "vocab": [("伯", "はく", "bác; bá tước; anh cả ."), ("伯仲", "はくちゅう", "sự ngang bằng; sự sánh kịp; sự bì kịp .")]},
    "箔": {"viet": "BẠC", "meaning_vi": "Rèm. Như châu bạc [珠箔] bức rèm châu. Bạch Cư Dị [白居易] : Lãm y thôi chẩm khởi bồi hồi, Châu bạc ngân bình dĩ lị khai [攬衣推枕起徘徊，珠箔銀屏迤邐開] (Trường hận ca [長恨歌]) Kéo áo lên, đẩy gối, bồi hồi trở dậy, Rèm châu bình bạc chầm chậm mở ra.", "meo": "", "reading": "すだれ", "vocab": [("アルミ箔", "アルミはく", "lá nhôm")]},
    "舶": {"viet": "BẠC", "meaning_vi": "Tàu buồm, thuyền lớn đi bể.", "meo": "", "reading": "ハク", "vocab": [("舶来", "はくらい", "nhập khẩu"), ("船舶", "せんぱく", "tàu thuỷ")]},
    "粕": {"viet": "PHÁCH", "meaning_vi": "Tao phách [糟粕] cặn rượu, bã giả. Phàm cái gì không có tinh túy đều gọi là tao phách.", "meo": "", "reading": "かす", "vocab": [("油粕", "あぶらかす", "bánh khô dầu .")]},
    "柏": {"viet": "BÁCH", "meaning_vi": "Biển bách [扁柏] cây biển bách. Một thứ cây to, dùng để đóng đồ.", "meo": "", "reading": "かしわ", "vocab": [("柏", "かしわ", "cây sồi"), ("柏木", "かしわぎ", "<THựC> gỗ sồi")]},
    "憂": {"viet": "ƯU", "meaning_vi": "Lo, buồn rầu.", "meo": "", "reading": "うれ.える うれ.い う.い う.き", "vocab": [("憂い", "うい", "buồn bã; lo lắng; u sầu; u uất"), ("憂い", "うれい", "nỗi đau buồn; nỗi thương tiếc; nỗi buồn")]},
    "腺": {"viet": "TUYẾN", "meaning_vi": "Trong thể xác các động vật chỗ nào bật chất nước ra gọi là tuyến. Như nhũ tuyến [乳腺] hạch sữa, hãn tuyến [汗腺] hạch mồ hôi.", "meo": "", "reading": "セン", "vocab": [("腺", "せん", "tuyến"), ("乳腺", "にゅうせん", "Tuyến vú .")]},
    "請": {"viet": "THỈNH, TÍNH", "meaning_vi": "Thăm hầu. Như thỉnh an [請安] hỏi thăm xem có được bình yên không.", "meo": "", "reading": "こ.う う.ける", "vocab": [("請い", "こい", "yêu cầu"), ("請う", "こう", "hỏi; yêu cầu; đề nghị; mời")]},
    "靖": {"viet": "TĨNH", "meaning_vi": "Yên bình, thái bình, làm yên", "meo": "Chữ XANH (青) đứng bên cạnh CHỐI BỎ (立) điều gì đó để mong cầu sự YÊN BÌNH.", "reading": "やす", "vocab": [("靖国神社", "やすくにじんじゃ", "Đền Yasukuni"), ("平靖", "へいせい", "Hòa bình, thái bình")]},
    "錆": {"viet": "", "meaning_vi": "tarnish", "meo": "", "reading": "さび くわ.しい", "vocab": [("錆", "さび", "gỉ; gỉ sét"), ("錆びる", "さびる", "gỉ; bị gỉ; mai một")]},
    "鯖": {"viet": "CHINH, THINH", "meaning_vi": "Cách nấu nướng. Cá nấu lẫn với thịt gọi là chinh. Lâu hộ nhà Hán [漢] từng đem các món ăn quý của Ngũ Hầu Vương Thị tặng nấu làm đồ ăn, đời gọi là ngũ hầu chinh [五侯鯖]. Cũng đọc là thinh.", "meo": "", "reading": "さば", "vocab": [("鯖", "さば", "cá thu; cá bạc má .")]},
    "揮": {"viet": "HUY", "meaning_vi": "Rung động, lay động. Như huy đao [揮刀] khoa đao, huy hào [揮毫] quẫy bút, v.v.", "meo": "", "reading": "ふる.う", "vocab": [("揮う", "ふるう", "dùng"), ("指揮", "しき", "chỉ huy")]},
    "輝": {"viet": "HUY", "meaning_vi": "Sáng sủa, rực rỡ. Làm nên vẻ vang gọi là quang huy [光輝]. Mạnh Giao [孟郊] : Thùy ngôn thốn thảo tâm, Báo đắc tam xuân huy [誰言寸草心, 報得三春輝] (Du tử ngâm [遊子吟]) ai nói rằng lòng của một tấc cỏ ngắn ngủi, hẹp hòi lại có thể báo đáp được ánh nắng ba mùa xuân chan hòa đầm ấm. Câu Liệu đem tấc cỏ quyết đền ba xuân của Nguyễn Du [阮攸] mượn ý hai câu thơ này.", "meo": "", "reading": "かがや.く", "vocab": [("輝き", "かがやき", "ánh sáng chói lọi"), ("輝く", "かがやく", "chói")]},
    "蓮": {"viet": "LIÊN", "meaning_vi": "Hoa sen. Con gái bó chân thon thon nên gọi là kim liên [金蓮]. Đông Hôn Hầu [東昏侯] chiều vợ, xây vàng làm hoa sen ở sân cho Phan Phi [潘妃] đi lên rồi nói rằng mỗi bước nẩy một đóa hoa sen. Vì thế nên gọi chân đàn bà là kim liên [金蓮].", "meo": "", "reading": "はす はちす", "vocab": [("蓮", "はす", "sen"), ("日蓮", "にちれん", "Nhật liên")]},
    "縺": {"viet": "", "meaning_vi": "tangle, knot, fasten, fetter", "meo": "", "reading": "もつ.れる", "vocab": [("縺れ", "もつれ", "tảo bẹ"), ("縺れる", "もつれる", "rối tung; lộn xộn")]},
    "斬": {"viet": "TRẢM", "meaning_vi": "Chém. Như trảm thảo [斬草] chém cỏ, trảm thủ [斬首] chém đầu, v.v.", "meo": "", "reading": "き.る", "vocab": [("斬る", "きる", "chém ."), ("斬新", "ざんしん", "mới")]},
    "暫": {"viet": "TẠM", "meaning_vi": "Chốc lát, không lâu. Như tạm thì [暫時].", "meo": "", "reading": "しばら.く", "vocab": [("暫く", "しばらく", "nhanh chóng; chốc lát; nhất thời; tạm thời; một lúc ."), ("暫定", "ざんてい", "sự tạm thời")]},
    "漸": {"viet": "TIỆM, TIÊM, TIỀM", "meaning_vi": "Sông Tiệm.", "meo": "", "reading": "ようや.く やや ようよ.う すす.む", "vocab": [("漸く", "ようやく", "một cách từ từ; một cách thong thả; dần dần"), ("漸増", "ぜんぞう", "sự tăng chậm chạp; sự tăng dần dần .")]},
    "陣": {"viet": "TRẬN", "meaning_vi": "Hàng trận, hàng lối quân lính. Cho nên chia bày đội quân gọi là trận.", "meo": "", "reading": "ジン", "vocab": [("陣", "じん", "trại"), ("一陣", "いちじん", "tiền đội")]},
    "軸": {"viet": "TRỤC", "meaning_vi": "Cái trục xe.", "meo": "", "reading": "ジク", "vocab": [("軸", "じく", "cán bút"), ("三軸", "さんじく", "ba trục")]},
    "軌": {"viet": "QUỸ", "meaning_vi": "Cái vết bánh xe chỗ trục hai bánh xe cách nhau. Đường sắt xe hỏa, xe điện chạy gọi là thiết quỹ [鐵軌] hay quỹ đạo [軌道].", "meo": "", "reading": "キ", "vocab": [("不軌", "ふき", "tình trạng không có pháp luật"), ("常軌", "じょうき", "sự thông thường; quỹ đạo thông thường")]},
    "腫": {"viet": "THŨNG, TRŨNG", "meaning_vi": "Sưng, nề. Như viêm thũng [炎腫] bệnh sưng lên vì nóng sốt.", "meo": "", "reading": "は.れる は.れ は.らす く.む はれもの", "vocab": [("腫", "しゅ", "khối u"), ("腫れ", "はれ", "sự phồng ra")]},
    "衝": {"viet": "XUNG", "meaning_vi": "va chạm, xung đột, xung kích", "meo": "Bộ \"hành\" (行) trên đầu chỉ hướng đi, \"trọng\" (重) chỉ sự nặng nề. Xung đột là sự va chạm mạnh hướng về phía trước.", "reading": "しょう", "vocab": [("衝突", "しょうとつ", "xung đột, va chạm"), ("衝撃", "しょうげき", "sốc, tác động, ảnh hưởng")]},
    "勲": {"viet": "HUÂN", "meaning_vi": "Công. Có công thưởng cho một cái dấu hiệu để tiêu biểu sự vẻ vang gọi là huân chương [勲章] như cái mền-đay bây giờ. Ngày xưa dùng chữ [勛], nay cũng thông dụng.", "meo": "", "reading": "いさお", "vocab": [("偉勲", "いくん", "thành công vĩ đại; thành tích vĩ đại"), ("勲功", "くんこう", "sự ban chức tước; sự phong sắc")]},
    "董": {"viet": "ĐỔNG", "meaning_vi": "Đốc trách. Như đổng sự [董事] giữ quyền đốc trách công việc.", "meo": "", "reading": "ただ.す", "vocab": [("骨董", "こっとう", "cổ"), ("骨董品", "こっとうひん", "vật hiếm có")]},
    "薫": {"viet": "HUÂN", "meaning_vi": "Thơm; đầm ấm; hơi khói", "meo": "", "reading": "かお.る", "vocab": [("薫り", "かおり", "hương thơm; hương vị; hơi hướng"), ("薫る", "かおる", "ngửi; tỏa hương")]},
    "萌": {"viet": "MANH", "meaning_vi": "Mầm cỏ, cây cỏ mới mọc đều gọi là manh nẩy mầm.", "meo": "", "reading": "も.える きざ.す めばえ きざ.し", "vocab": [("萌し", "きざし", "sự mọc mầm; đâm chồi; nảy chồi"), ("萌芽", "ほうが", "sự mọc mộng")]},
    "宰": {"viet": "TỂ", "meaning_vi": "Chúa tể. Như tâm giả đạo chi chủ tể [心者道之主宰] tâm là cái chúa tể của đạo.", "meo": "", "reading": "サイ", "vocab": [("主宰", "しゅさい", "sự chủ tọa; sự tổ chức"), ("宰相", "さいしょう", "thủ tướng .")]},
    "執": {"viet": "CHẤP", "meaning_vi": "Cầm.", "meo": "", "reading": "と.る", "vocab": [("執る", "とる", "cầm lấy"), ("執事", "しつじ", "người quản lý")]},
    "摯": {"viet": "CHÍ", "meaning_vi": "Rất, lắm. Như ái chi thậm chí [愛之甚摯] yêu chuộng rất mực.", "meo": "", "reading": "いた.る", "vocab": [("真摯", "しんし", "tính thành thật")]},
    "拙": {"viet": "CHUYẾT", "meaning_vi": "Vụng về.", "meo": "", "reading": "つたな.い", "vocab": [("拙い", "つたない", "vụng"), ("拙劣", "せつれつ", "sự vụng về; sự không khéo léo")]},
    "堀": {"viet": "QUẬT", "meaning_vi": "Hang, động. Như chữ [窟].", "meo": "", "reading": "ほり", "vocab": [("堀", "ほり", "hào (vây quanh thanh trì...); kênh đào"), ("堀割", "ほりわり", "kênh; sông đào; mương; hào .")]},
    "窟": {"viet": "QUẬT", "meaning_vi": "Cái hang, cái hang của giống thú ở gọi là quật. Nguyễn Trãi [阮廌] : Long Đại kim quan thạch quật kỳ [龍袋今觀石窟奇] (Long Đại nham [龍袋岩]) Nay xem ở Long Đại có hang đá kỳ lạ.", "meo": "", "reading": "いわや いはや あな", "vocab": [("偏窟", "へんくつ", "tính lập dị"), ("岩窟", "がんくつ", "Hang; hang động; hang đá")]},
    "尚": {"viet": "THƯỢNG", "meaning_vi": "Ngõ hầu. Như thượng hưởng [尚饗] ngõ hầu hưởng cho.", "meo": "", "reading": "なお", "vocab": [("尚", "なお", "chưa"), ("今尚", "いまなお", "im")]},
    "償": {"viet": "THƯỜNG", "meaning_vi": "Đền, bù. Như thường hoàn [償還] đền trả, đắc bất thường thất [得不償失] số được chẳng bù số mất.", "meo": "", "reading": "つぐな.う", "vocab": [("償い", "つぐない", "sự thưởng"), ("償う", "つぐなう", "bồi thường")]},
    "掌": {"viet": "CHƯỞNG", "meaning_vi": "Lòng bàn tay, quyền ở trong tay gọi là chưởng ác chi trung [掌握之中].", "meo": "", "reading": "てのひら たなごころ", "vocab": [("掌", "たなごころ", "gan bàn tay"), ("掌", "てのひら", "lòng bàn tay .")]},
    "嘗": {"viet": "THƯỜNG", "meaning_vi": "Nếm. Lễ ký [禮記] : Quân hữu tật, ẩm dược, thần tiên thường chi [君有疾, 飲藥, 臣先嘗之] (Khúc lễ hạ [曲禮下]) Nhà vua có bệnh, uống thuốc, bầy tôi nếm trước.", "meo": "", "reading": "かつ.て こころ.みる な.める", "vocab": [("嘗て", "かつて", "đã có một thời; đã từng; trước kia"), ("嘗める", "なめる", "cái liềm")]},
    "樋": {"viet": "", "meaning_vi": "water pipe, gutter, downspout, conduit", "meo": "", "reading": "ひ とい", "vocab": [("樋", "とい", "ống nước .")]},
    "桶": {"viet": "DŨNG", "meaning_vi": "Cái thùng gỗ hình tròn. Như thủy dũng [水桶] thùng nước.", "meo": "", "reading": "おけ", "vocab": [("桶", "おけ", "cái xô; xô đựng nước"), ("手桶", "ておけ", "Cái xô; cái thùng")]},
    "猟": {"viet": "LIỆP", "meaning_vi": "Săn bắn", "meo": "", "reading": "かり か.る", "vocab": [("猟", "りょう", "săn"), ("猟人", "かりゅうど", "Người đi săn; thợ săn")]},
    "蝋": {"viet": "", "meaning_vi": "wax", "meo": "", "reading": "みつろう ろうそく", "vocab": [("蝋", "ろう", "sáp ong"), ("屍蝋", "しろう", "chất sáp mỡ")]},
    "酷": {"viet": "KHỐC", "meaning_vi": "Tàn ác. Như khốc lại [酷吏] quan lại tàn ác.", "meo": "", "reading": "ひど.い", "vocab": [("酷", "こく", "khắc khe"), ("酷い", "ひどい", "kinh khủng; khủng khiếp")]},
    "縁": {"viet": "DUYÊN", "meaning_vi": "Duyên số, số mệnh.", "meo": "", "reading": "ふち ふちど.る ゆかり よすが へり えにし", "vocab": [("縁", "えん", "duyên; duyên nợ; nghiệp chướng; giao tình"), ("縁", "ふち", "mép; lề; viền")]},
    "嫁": {"viet": "GIÁ", "meaning_vi": "Lấy chồng. Kinh Lễ định con gái hai mươi tuổi thì lấy chồng gọi là xuất giá [出嫁].", "meo": "", "reading": "よめ とつ.ぐ い.く ゆ.く", "vocab": [("嫁", "よめ", "cô dâu"), ("嫁ぐ", "とつぐ", "cưới")]},
    "逐": {"viet": "TRỤC", "meaning_vi": "Đuổi, đuổi theo. Như truy trục [追逐] đuổi theo.", "meo": "", "reading": "チク", "vocab": [("逐一", "ちくいち", "cụ thể; chi tiết; nhất nhất từng việc"), ("逐年", "ちくねん", "hàng năm")]},
    "遂": {"viet": "TOẠI", "meaning_vi": "Thỏa thích. Như toại chí [遂志] thích chí. Bất toại sở nguyện [不遂所願] không được thỏa nguyện.", "meo": "", "reading": "と.げる つい.に", "vocab": [("遂に", "ついに", "cuối cùng"), ("完遂", "かんすい", "sự hoàn thành; hoàn thành")]},
    "墜": {"viet": "TRỤY", "meaning_vi": "Rơi, rụng. Nguyễn Du [阮攸] : Phạt tận tùng chi trụy hạc thai [伐盡松枝墜鶴胎] (Vọng Quan Âm miếu [望觀音廟]) Chặt hết cành tùng, rớt trứng hạc.", "meo": "", "reading": "お.ちる お.つ", "vocab": [("失墜", "しっつい", "sự mất (quyền uy"), ("墜ちる", "おちる", "giọt (nước")]},
    "豪": {"viet": "HÀO", "meaning_vi": "Con hào, một loài thú như loài lợn.", "meo": "", "reading": "えら.い", "vocab": [("豪", "ごう", "ào ạt; to; lớn xối xả; như trút nước"), ("豪い", "えらい", "lớn")]},
    "壕": {"viet": "HÀO", "meaning_vi": "Cái hào.", "meo": "", "reading": "ほり", "vocab": [("壕", "ごう", "hầm hố ."), ("塹壕", "ざんごう", "rãnh")]},
    "蒙": {"viet": "MÔNG", "meaning_vi": "Tối. Chỗ mặt trời lặn gọi là đại mông [大蒙].", "meo": "", "reading": "こうむ.る おお.う くら.い", "vocab": [("蒙", "こうむ", "sự ngu dốt"), ("蒙る", "こうむる", "nhận")]},
    "鍵": {"viet": "KIỆN", "meaning_vi": "Cái khóa, cái lá mía khóa.", "meo": "", "reading": "かぎ", "vocab": [("鍵", "かぎ", "chốt"), ("打鍵", "だけん", "gõ phím")]},
    "吉": {"viet": "CÁT", "meaning_vi": "Tốt lành. Phàm việc gì vui mừng đều gọi là cát [吉], đối lại với chữ hung [凶]. Như cát tường [吉祥] điềm lành.", "meo": "", "reading": "よし", "vocab": [("不吉", "ふきつ", "chẳng lành; bất hạnh; không may"), ("吉兆", "きっちょう", "điềm lành; may; may mắn")]},
    "舎": {"viet": "XÁ", "meaning_vi": "Cư xá", "meo": "", "reading": "やど.る", "vocab": [("舎", "しゃ", "chuồng"), ("舎兄", "しゃけい", "chứng vẹo cổ")]},
    "只": {"viet": "CHÍCH, CHỈ", "meaning_vi": "Nhời trợ ngữ. Như lạc chỉ quân tử [樂只君子] vui vậy người quân tử.", "meo": "", "reading": "ただ", "vocab": [("只", "ただ", "chỉ; đơn thuần"), ("只々", "ただ々", "tuyệt đối")]},
    "克": {"viet": "KHẮC", "meaning_vi": "Hay. Như bất khắc thành hành [不克成行] không hay đi được.", "meo": "", "reading": "か.つ", "vocab": [("克己", "こっき", "khắc kỵ"), ("克復", "こくふく", "sự hoàn lại")]},
    "呪": {"viet": "CHÚ", "meaning_vi": "Nguyền rủa.", "meo": "", "reading": "まじな.う のろ.い まじな.い のろ.う", "vocab": [("呪い", "のろい", "lời nguyền rủa"), ("呪う", "のろう", "nguyền rủa")]},
    "卓": {"viet": "TRÁC", "meaning_vi": "Cao chót. Như trác thức [卓識] kiến thức cao hơn người, trác tuyệt [卓絕] tài trí tuyệt trần.", "meo": "", "reading": "タク", "vocab": [("卓", "たく", "cái bàn"), ("卓上", "たくじょう", "để bàn")]},
    "悼": {"viet": "ĐIỆU", "meaning_vi": "Thương.", "meo": "", "reading": "いた.む", "vocab": [("悼む", "いたむ", "chia buồn; đau buồn"), ("哀悼", "あいとう", "lời chia buồn")]},
    "悦": {"viet": "DUYỆT", "meaning_vi": "Giản thể của chữ 悅", "meo": "", "reading": "よろこ.ぶ よろこ.ばす", "vocab": [("悦", "えつ", "sự tự mãn; mãn nguyện; sung sướng"), ("悦び", "よろこび", "sự sung sướng vô ngần")]},
    "閲": {"viet": "DUYỆT", "meaning_vi": "Dị dạng của chữ 阅", "meo": "", "reading": "けみ.する", "vocab": [("閲兵", "えっぺい", "sự phô trương"), ("査閲", "さえつ", "sự xem xét kỹ")]},
    "鯉": {"viet": "LÍ", "meaning_vi": "Cá chép.", "meo": "", "reading": "こい", "vocab": [("鯉", "こい", "cá chép")]},
    "狸": {"viet": "LI", "meaning_vi": "Con li, một loài như loài hồ.", "meo": "", "reading": "たぬき", "vocab": [("狸", "たぬき", "con lửng"), ("古狸", "ふるだぬき", "người kỳ cựu")]},
    "厘": {"viet": "LI, HI", "meaning_vi": "Cũng như chữ [釐].", "meo": "", "reading": "リン", "vocab": [("厘", "りん", "linh"), ("厘毛", "りんもう", "món tiền nhỏ .")]},
    "糧": {"viet": "LƯƠNG", "meaning_vi": "Thức ăn, lương ăn. Thức ăn lúc đi đường gọi là lương [糧], lúc ở ngay nhà gọi là thực [食]. Nay gọi các vật dùng trong quân là lương.", "meo": "", "reading": "かて", "vocab": [("兵糧", "ひょうろう", "lương của quân đội; lương thảo"), ("心の糧", "こころのかて", "món ăn tinh thần")]},
    "瞳": {"viet": "ĐỒNG", "meaning_vi": "Lòng tử, con ngươi.", "meo": "", "reading": "ひとみ", "vocab": [("瞳", "ひとみ", "con ngươi"), ("瞳子", "どうし", "học trò")]},
    "鐘": {"viet": "CHUNG", "meaning_vi": "Cái chuông. Trong chùa cứ sớm và tối thì khua chuông, cho nên mới gọi cái đồng hồ đánh chuông là chung. Trần Nhân Tông [陳仁宗] : Ngư thuyền tiêu sắt mộ chung sơ [漁船蕭瑟暮鐘初] (Lạng Châu vãn cảnh [諒州晚景]) Chiếc thuyền đánh cá trong tiếng chuông chiều buồn bã vừa điểm.", "meo": "", "reading": "かね", "vocab": [("鐘", "かね", "chuông ."), ("半鐘", "はんしょう", "chuông dùng để báo hỏa họan .")]},
    "憧": {"viet": "SUNG, TRÁNG", "meaning_vi": "Sung sung [憧憧] lông bông ý chưa định hẳn cứ lông bông hoài gọi là sung sung.", "meo": "", "reading": "あこが.れる", "vocab": [("憧れ", "あこがれ", "niềm mơ ước"), ("憧れる", "あこがれる", "mong ước; mơ ước")]},
    "纏": {"viet": "TRIỀN", "meaning_vi": "Ràng rịt, quấn quanh, vây bọc. Như triền nhiễu [纏繞] chèn chặn, triền túc [纏足] bó chân (tục cổ Trung Hoa); triền miên [纏綿] quấn quýt.", "meo": "", "reading": "まつ.わる まと.う まと.める まと.まる まと.い", "vocab": [("纏う", "まとう", "sự mang; sự dùng; sự mặc"), ("纏め", "まとめ", "kết luận; sự kết luận")]},
    "墨": {"viet": "MẶC", "meaning_vi": "Sắc đen.", "meo": "", "reading": "すみ", "vocab": [("墨", "すみ", "mực; mực đen"), ("入墨", "いれずみ", "hiệu trống tập trung buổi tối")]},
    "黙": {"viet": "MẶC", "meaning_vi": "Trầm mặc,", "meo": "", "reading": "だま.る もだ.す", "vocab": [("黙々", "もくもく", "không nói"), ("黙り", "だんまり", "sự lặng thinh")]},
    "猪": {"viet": "TRƯ", "meaning_vi": "Tục dùng như chữ trư [豬].", "meo": "", "reading": "い いのしし", "vocab": [("猪", "いのしし", "heo rừng"), ("猪突", "ちょとつ", "tính không lo lắng")]},
    "奢": {"viet": "XA", "meaning_vi": "Xa xỉ. Lý Thương Ẩn [李商隱] : Thành do cần kiệm phá do xa [成由勤儉破由奢] (Vịnh sử [詠史]) Nên việc là do cần kiệm, đổ vỡ vì hoang phí.", "meo": "", "reading": "おご.る おご.り", "vocab": [("奢る", "おごる", "chăm sóc; chiêu đãi; khoản đãi; khao ."), ("奢侈", "しゃし", "sự xa xỉ")]},
    "曙": {"viet": "THỰ", "meaning_vi": "Rạng đông, sáng. Bạch Cư Dị [白居易] : Cảnh cảnh tinh hà dục thự thiên [耿耿星河欲曙天] (Trường hận ca [長恨歌] ) Những ngôi sao trên sông ngân (tinh hà) sáng lấp lánh như muốn là rạng đông.", "meo": "", "reading": "あけぼの", "vocab": [("曙", "あけぼの", "Hửng sáng; lúc rạng đông; rạng đông; bắt đầu một ngày mới"), ("曙光", "しょこう", "bình minh")]},
    "煮": {"viet": "CHỬ", "meaning_vi": "Nấu, thổi. Như chử phạn [煮飯] nấu cơm. Đặng Trần Côn [鄧陳琨] : Tạ sầu hề vi chẩm, Chử muộn hề vi xan  [藉愁兮爲枕, 煮悶兮爲餐] (Chinh Phụ ngâm [征婦吟]) Tựa sầu làm gối, Nấu muộn làm cơm. Đoàn Thị Điểm dịch thơ : Sầu ôm nặng, hãy chồng làm gối, Buồn chứa đầy, hãy thổi làm cơm. $ Xem [煑].", "meo": "", "reading": "に.る -に に.える に.やす", "vocab": [("煮る", "にる", "nấu"), ("煮干", "にぼし", "Cá mòi khô (thường dùng để nấu món súp MISO)")]},
    "躇": {"viet": "TRỪ", "meaning_vi": "Trù trừ [躊躇] do dự, rụt rè.", "meo": "", "reading": "ためら.う", "vocab": [("躊躇", "ちゅうちょ", "Sự ngập ngừng; sự do dự ."), ("躊躇い", "ためらい", "ấp úng")]},
    "箸": {"viet": "TRỨ, TRỢ", "meaning_vi": "Cái đũa, cùng nghĩa với chữ khoái [筷]. Ta quen đọc là chữ trợ. Nguyễn Du [阮攸] : Mãn trác trần trư dương, Trưởng quan bất hạ trợ [滿棹陳豬羊, 長官不下箸] (Sở kiến hành [所見行]) Đầy bàn thịt heo, thịt dê, Quan lớn không đụng đũa.", "meo": "", "reading": "はし", "vocab": [("箸", "はし", "đũa ."), ("匕箸", "ひちょ", "Thìa và đũa .")]},
    "賭": {"viet": "ĐỔ", "meaning_vi": "Đánh bạc, cờ bạc.", "meo": "", "reading": "か.ける かけ", "vocab": [("賭", "と", "sự đánh cuộc"), ("賭け", "かけ", "trò cá cược; trò cờ bạc; việc chơi cờ bạc ăn tiền")]},
    "諸": {"viet": "CHƯ", "meaning_vi": "Chưng, có ý nghĩa nói chuyện về một chỗ. Như quân tử cầu chư kỉ [君子求諸己] (Luận ngữ [論語]) người quân tử chỉ cầu ở mình.", "meo": "", "reading": "もろ", "vocab": [("諸", "しょ", "các; nhiều; vài ."), ("諸々", "もろもろ", "khác nhau; nhiều thứ khác nhau")]},
    "儲": {"viet": "TRỮ, TRỪ", "meaning_vi": "Trữ, tích chứa, để dành.", "meo": "", "reading": "もう.ける もう.かる もうけ たくわ.える", "vocab": [("儲け", "もうけ", "lợi nhuận; tiền lãi ."), ("一儲", "ひともうけ", "Sự đúc tiền .")]},
    "玩": {"viet": "NGOẠN", "meaning_vi": "Vờn, chơi, đùa bỡn. Như ngoạn nhân táng đức, ngoạn vật táng chí [玩人喪德，玩物喪志] (Thư Kinh [書經]) đùa bỡn người hỏng đức, vờn chơi vật hỏng chí. Những đồ để ngắm chơi gọi là ngoạn cụ [玩具], đồ chơi quý gọi là trân ngoạn [珍玩].", "meo": "", "reading": "もちあそ.ぶ もてあそ.ぶ", "vocab": [("玩具", "おもちゃ", "đồ chơi"), ("玩味", "がんみ", "đồ gia vị (nước xốt")]},
    "冠": {"viet": "QUAN, QUÁN", "meaning_vi": "Cái mũ.", "meo": "", "reading": "かんむり", "vocab": [("冠", "かんむり", "mũ miện; vương miện"), ("お冠", "おかんむり", "sự cực kỳ tức giận; sự vô cùng tức giận; sự tức tối tột độ")]},
    "姑": {"viet": "CÔ", "meaning_vi": "Mẹ chồng.", "meo": "", "reading": "しゅうとめ しゅうと おば しばらく", "vocab": [("姑", "しゅうとめ", "mẹ chồng ."), ("小姑", "こじゅうと", "chị dâu")]},
    "故": {"viet": "CỐ", "meaning_vi": "Việc. Như đại cố [大故] việc lớn, đa cố [多故] lắm việc, v.v.", "meo": "", "reading": "ゆえ ふる.い もと", "vocab": [("故", "こ", "cố; cũ (đi ghép với từ khác)"), ("故", "ゆえ", "lý do; nguyên nhân; nguồn cơn .")]},
    "箇": {"viet": "CÁ", "meaning_vi": "đếm (vật nhỏ)", "meo": "Vật nhỏ (固) được quấn vào (个) để đếm 1 CÁ (箇).", "reading": "か", "vocab": [("箇所", "かしょ", "chỗ, địa điểm, nơi"), ("一個", "いっこ", "một cái")]},
    "錮": {"viet": "CỐ", "meaning_vi": "Hàn, dùng các thứ đồng thứ sắt hàn bịt các lỗ các khiếu đều gọi là cố.", "meo": "", "reading": "ふさ.ぐ", "vocab": [("禁錮", "きんこ", "sự giam")]},
    "胡": {"viet": "HỒ", "meaning_vi": "Yếm cổ, dưới cổ có mảng thịt sa xuống gọi là hồ. Râu mọc ở đấy gọi là hồ tu [胡鬚]. Tục viết là [鬍].", "meo": "", "reading": "なんぞ", "vocab": [("胡乱", "うろん", "cá; có mùi cá"), ("胡座", "あぐら", "kiểu ngồi khoanh chân; kiểu ngồi thiền; ngồi thiền; thiền; ngồi xếp bằng tròn")]},
    "湖": {"viet": "HỒ", "meaning_vi": "Cái hồ.", "meo": "", "reading": "みずうみ", "vocab": [("湖", "みずうみ", "hồ ."), ("塩湖", "しおみずうみ", "sự kể lại")]},
    "糊": {"viet": "HỒ", "meaning_vi": "Hồ dính, hồ để dán.", "meo": "", "reading": "のり", "vocab": [("糊", "のり", "hồ dán; hồ vải; bột hồ; keo dán ."), ("糊口", "ここう", "sự tồn tại")]},
    "瑚": {"viet": "HÔ, HỒ", "meaning_vi": "San hô [珊瑚] một thứ động vật nhỏ ở trong bể kết lại, hình như cành cây, đẹp như ngọc, dùng làm chỏm mũ rất quý.", "meo": "", "reading": "コ ゴ", "vocab": [("珊瑚", "さんご", "san hô"), ("珊瑚礁", "さんごしょう", "bãi san hô")]},
    "据": {"viet": "CƯ, CỨ", "meaning_vi": "Kiết cư [拮据] bệnh tay.", "meo": "", "reading": "す.える す.わる", "vocab": [("据える", "すえる", "đặt"), ("据わる", "すわる", "ngồi xổm")]},
    "裾": {"viet": "CƯ, CỨ", "meaning_vi": "Vạt áo.", "meo": "", "reading": "すそ", "vocab": [("裾", "すそ", "tà áo"), ("お裾分け", "おすそわけ", "sự phân chia; sự phân bổ")]},
    "粘": {"viet": "NIÊM", "meaning_vi": "Tục dùng như chữ niêm [黏]. Nguyễn Du [阮攸] : Tạc kiến tân trịnh thành môn niêm bảng thị [昨見新鄭城門粘榜示] (Trở binh hành [阻兵行]) Hôm trước thấy cửa thành Tân Trịnh yết bảng cáo thị.", "meo": "", "reading": "ねば.る", "vocab": [("粘々", "ねばねば", "sự dính; sự dinh dính ."), ("粘い", "ねばい", "dính; sánh; bầy nhầy")]},
    "貼": {"viet": "THIẾP", "meaning_vi": "Phụ thêm bù thêm vào chỗ thiếu gọi là thiếp. Như tân thiếp [津貼] thấm thêm, giúp thêm.", "meo": "", "reading": "は.る つ.く", "vocab": [("貼る", "はる", "dán; gắn cho"), ("下貼", "したは", "áo bành tô mặc trong")]},
    "帖": {"viet": "THIẾP", "meaning_vi": "Lấy lụa viết chữ vào lụa. Đời xưa chưa có giấy, phải viết vào lụa gọi là thiếp. Đời sau viết vào giấy cũng gọi là thiếp. Như xuân thiếp [春帖] câu đối tết, phủ thiếp [府帖] dấu hiệu làm tin trong quan tràng, giản thiếp [柬帖] cái danh thiếp, nê kim thiếp tử [泥金帖子] cái đơn hàng hay nhãn hiệu xoa kim nhũ, v.v.", "meo": "", "reading": "かきもの", "vocab": [("手帖", "てちょう", "Sổ tay ."), ("法帖", "ほうじょう", "sự tốt")]},
    "鮎": {"viet": "NIÊM", "meaning_vi": "Cá niêm, cá măng. Mình tròn mà dài, đầu to đuôi dẹt, không có vẩy, nhiều chất dính, mồm cong mà rộng, hai bên hàm mọc răng nanh nhỏ, có râu, lưng xanh đen, bụng trắng, có con lớn dài đến hai thước (Parasilurus asotus).", "meo": "", "reading": "あゆ なまず", "vocab": [("鮎", "あゆ", "cá chẻm .")]},
    "彫": {"viet": "ĐIÊU", "meaning_vi": "Chạm trổ.", "meo": "", "reading": "ほ.る -ぼ.り", "vocab": [("彫る", "ほる", "cẩn"), ("彫像", "ちょうぞう", "bức tượng .")]},
    "鯛": {"viet": "ĐIÊU", "meaning_vi": "Cá điêu. Tục gọi là đồng bồn ngư [銅盆魚].", "meo": "", "reading": "たい", "vocab": [("鯛", "たい", "cá hồng ."), ("黒鯛", "くろだい", "Cá tráp biển đen .")]},
    "苛": {"viet": "HÀ", "meaning_vi": "Nghiệt ác. Làm việc xét nét nghiêm ngặt quá đều gọi là hà. Chánh lệnh tàn ác gọi là hà chánh [苛政]. Lễ ký [禮記] : Hà chánh mãnh ư hổ dã [苛政猛於虎也] chính sách hà khắc còn tàn bạo hơn cọp", "meo": "", "reading": "いじ.める さいな.む いらだ.つ からい こまかい", "vocab": [("苛々", "イライラ", "sốt ruột; nóng ruột"), ("苛々", "いらいら", "sự sốt ruột; sự nóng ruột; tức giận; khó chịu")]},
    "阿": {"viet": "A, Á", "meaning_vi": "Cái đống lớn, cái gò to.", "meo": "", "reading": "おもね.る くま", "vocab": [("阿る", "おもねる", "tâng bốc"), ("阿吽", "あうん", "vt của Order of Merit")]},
    "扇": {"viet": "PHIẾN, THIÊN", "meaning_vi": "Cánh cửa.", "meo": "", "reading": "おうぎ", "vocab": [("扇", "おうぎ", "quạt gấp; quạt giấy; quạt"), ("扇ぐ", "あおぐ", "quạt")]},
    "詔": {"viet": "CHIẾU", "meaning_vi": "Ban bảo, dẫn bảo. Ngày xưa người trên bảo kẻ dưới là chiếu, từ nhà Tần [秦] nhà Hán [漢] trở xuống thì chỉ vua được dùng thôi. Như chiếu thư [詔書] tờ chiếu, ân chiếu [恩詔] xuống chiếu ra ơn cho, v. v.", "meo": "", "reading": "みことのり", "vocab": [("詔", "みことのり", "chiếu chỉ; mệnh lệnh của thiên hoàng ."), ("詔勅", "しょうちょく", "chiếu chỉ; văn bản biểu thị ý chí của thiên hoàng .")]},
    "岳": {"viet": "NHẠC", "meaning_vi": "Cũng như chữ nhạc [嶽]. Như Ngũ Nhạc [五岳] năm núi Nhạc : Tung Sơn [嵩山], Thái Sơn [泰山], Hoa Sơn [華山], Hành Sơn [衡山], Hằng Sơn [恆山]. Trên Thái Sơn có một ngọn núi là trượng nhân phong [丈人峯]. Vì thế nên bố vợ gọi là nhạc trượng [岳丈]. Tục dùng chữ nhạc này cả.", "meo": "", "reading": "たけ", "vocab": [("岳", "たけ", "núi cao ."), ("岳人", "たけひと", "người leo núi")]},
    "功": {"viet": "CÔNG", "meaning_vi": "Việc. Như nông công [農功] việc làm ruộng.", "meo": "", "reading": "いさお", "vocab": [("功利", "こうり", "sự có ích; tính có ích"), ("功労", "こうろう", "công lao; công trạng; sự đóng góp lớn lao")]},
    "貢": {"viet": "CỐNG", "meaning_vi": "Cống, dâng. Như tiến cống [進貢] dâng các vật thổ sản.", "meo": "", "reading": "みつ.ぐ", "vocab": [("貢ぎ", "みつぎ", "vật triều cống; đồ cống; đồ cống nạp ."), ("貢ぐ", "みつぐ", "trợ giúp (tài chính); giúp đỡ (tiền bạc)")]},
    "虹": {"viet": "HỒNG", "meaning_vi": "Cái cầu vồng. Nguyễn Du [阮攸] : Bạch hồng quán nhật thiên man man [白虹貫日天漫漫] (Kinh Kha cố lý [荊軻故里]) Cầu vồng trắng vắt ngang mặt trời, bầu trời mênh mang.", "meo": "", "reading": "にじ", "vocab": [("虹", "にじ", "cầu vồng"), ("虹彩", "こうさい", "Tròng đen; mống mắt")]},
    "項": {"viet": "HẠNG", "meaning_vi": "Cổ sau, gáy. Không chịu cúi đầu nhún lòng theo với người khác gọi là cường hạng [強項] cứng cổ.", "meo": "", "reading": "うなじ", "vocab": [("項", "こう", "mục; khoản; số hạng"), ("一項", "いちこう", "khoản")]},
    "控": {"viet": "KHỐNG", "meaning_vi": "Dẫn, kéo. Như khống huyền [控弦] dương cung.", "meo": "", "reading": "ひか.える ひか.え", "vocab": [("控え", "ひかえ", "lời ghi"), ("控える", "ひかえる", "chế ngự; kiềm chế; giữ gìn (lời ăn tiếng nói); điều độ (ăn uống)")]},
    "腔": {"viet": "KHANG, XOANG", "meaning_vi": "Xương rỗng. Các loài động vật như sâu bọ, san hô gọi là khang tràng động vật [腔腸動物] loài động vật ruột rỗng.", "meo": "", "reading": "コウ", "vocab": [("口腔", "こうくう", "khoang miệng"), ("満腔", "まんこう", "chân thành")]},
    "銘": {"viet": "MINH", "meaning_vi": "Bài minh. Khắc chữ vào đồ, hoặc để tự răn mình, hoặc ghi chép công đức gọi là minh. Ngày xưa khắc vào cái chuông cái đỉnh, đời sau hay khắc vào bia. Tọa hữu minh [座右銘], Thôi Viện [崔瑗] đời Đông Hán làm bài minh để bên phải chỗ ngồi của minh. Nguyễn Trãi [阮廌] : Hỉ đắc tân thi đáng tọa minh [喜得新詩當座銘] (Thứ vận Hoàng môn thị lang [次韻黃門侍郎]) Mừng được bài thơ mới đáng khắc làm bài minh để (bên phải) chỗ ngồi.", "meo": "", "reading": "メイ", "vocab": [("銘々", "めいめい", "mỗi người; mỗi cá thể"), ("感銘", "かんめい", "cảm động sâu sắc; sự nhớ đời; vô cùng cảm động; cảm động; xúc động; cảm kích")]},
    "妙": {"viet": "DIỆU", "meaning_vi": "Khéo, hay, thần diệu lắm. Tinh thần khéo léo mầu nhiệm không thể nghĩ nghị được gọi là diệu. Như diệu lí [妙理] lẽ huyền diệu.", "meo": "", "reading": "たえ", "vocab": [("妙", "みょう", "kỳ lạ; không bình thường"), ("妙味", "みょうみ", "thanh")]},
    "沙": {"viet": "SA, SÁ", "meaning_vi": "Cát.", "meo": "", "reading": "すな よなげる", "vocab": [("沙汰", "さた", "việc"), ("沙漠", "さばく", "công lao")]},
    "炒": {"viet": "SAO", "meaning_vi": "Sao, rang.", "meo": "", "reading": "い.る いた.める", "vocab": [("炒める", "いためる", "rán giòn; phi (hành mỡ)"), ("油で炒める", "あぶらでいためる", "chiên .")]},
    "抄": {"viet": "SAO", "meaning_vi": "Lấy qua. Tục gọi sự tịch kí nhà cửa là sao gia [抄家].", "meo": "", "reading": "ショウ", "vocab": [("抄出", "しょうしゅつ", "sự trích"), ("手抄", "しゅしょう", "sự trích")]},
    "劣": {"viet": "LIỆT", "meaning_vi": "Kém, đối lại với chữ ưu [優] hơn.", "meo": "", "reading": "おと.る", "vocab": [("劣る", "おとる", "kém hơn; thấp kém"), ("下劣", "げれつ", "cơ sở")]},
    "賓": {"viet": "TÂN, THẤN", "meaning_vi": "Khách, người ở ngoài đến gọi là khách [客], kính mời ngồi trên gọi là tân [賓]. Như tương kính như tân [相敬如賓] cùng kính nhau như khách quý. Ngày xưa đặt ra năm lễ, trong đó có một lễ gọi là tân lễ [賓禮], tức là lễ phép khách khứa đi lại thù tạc với nhau. Âu Dương Tu [歐陽修] : Chúng tân hoan dã [眾賓歡也] (Túy Ông đình ký [醉翁亭記]) Khách khứa vui thích vậy.", "meo": "", "reading": "ヒン", "vocab": [("賓客", "ひんきゃく", "khách mời danh dự ."), ("賓客", "ひんかく", "khách mời danh dự")]},
    "訣": {"viet": "QUYẾT", "meaning_vi": "Quyết biệt, sắp đi xa lâu mà tặng bằng lời gọi là quyết. Lời nói của kẻ chết trối lại gọi là lời vĩnh quyết [永訣].", "meo": "", "reading": "わかれ わかれ.る", "vocab": [("訣別", "けつべつ", "sự chia ly"), ("秘訣", "ひけつ", "bí quyết")]},
    "訟": {"viet": "TỤNG", "meaning_vi": "Kiện tụng, đem nhau lên quan mà tranh biện phải trái gọi là tụng. Như tranh tụng [爭訟] thưa kiện nhau, tố tụng [訴訟] cáo kiện.", "meo": "", "reading": "ショウ", "vocab": [("争訟", "そうしょう", "hay cãi nhau"), ("訴訟", "そしょう", "sự kiện tụng; sự tranh chấp; sự kiện cáo")]},
    "聡": {"viet": "THÔNG", "meaning_vi": "Tục dùng như chữ thông [聰].", "meo": "", "reading": "さと.い みみざと.い", "vocab": [("聡敏", "さとさとし", "sự thông minh"), ("聡明", "そうめい", "tính khôn ngoan")]},
    "翁": {"viet": "ÔNG", "meaning_vi": "Cha, mình gọi cha người khác, gọi là tôn ông [尊翁].", "meo": "", "reading": "おきな", "vocab": [("翁", "おう", "ông già; cụ già"), ("老翁", "ろうおう", "người đàn ông già cả .")]},
    "恭": {"viet": "CUNG", "meaning_vi": "Cung kính. Sự kính đã tỏ lộ ra ngoài gọi là cung [恭].", "meo": "", "reading": "うやうや.しい", "vocab": [("恭倹", "きょうけん", "sự chiều ý"), ("允恭", "まこときょう", "sự lịch sự")]},
    "翼": {"viet": "DỰC", "meaning_vi": "Cánh chim, chỗ cánh mọc lông ra để bay gọi là dực. Cánh của các loài sâu cũng gọi là dực.", "meo": "", "reading": "つばさ", "vocab": [("翼", "つばさ", "cánh"), ("翼々", "つばさ々", "thận trọng")]},
    "糞": {"viet": "PHẨN", "meaning_vi": "Phân, cứt. Như điểu phẩn [鳥糞] cứt chim, ngưu phẩn [牛糞] cứt bò.", "meo": "", "reading": "くそ", "vocab": [("糞", "くそ", "phân; cứt"), ("糞便", "ふんべん", "cặn")]},
    "智": {"viet": "TRÍ", "meaning_vi": "Khôn, trái với chữ ngu [愚], hiểu thấu sự lý gọi là trí.", "meo": "", "reading": "チ", "vocab": [("上智", "じょうち", "lâu dài"), ("人智", "じんち", "sự biết")]},
    "痴": {"viet": "SI", "meaning_vi": "Tục dùng như chữ si [癡].", "meo": "", "reading": "し.れる おろか", "vocab": [("痴人", "ちじん", "người ngớ ngẩn; thằng ngốc"), ("痴呆", "ちほう", "chứng mất trí")]},
    "疾": {"viet": "TẬT", "meaning_vi": "Ốm, tật bệnh, mình mẩy không được dễ chịu gọi là tật [疾], nặng hơn nữa gọi là bệnh [病].", "meo": "", "reading": "はや.い", "vocab": [("疾呼", "しっこ", "tiếng kêu; sự la hét"), ("廃疾", "はいしつ", "sự ốm yếu; tàn tật")]},
    "嫉": {"viet": "TẬT", "meaning_vi": "Ghen ghét, thấy người ta đức hạnh tài trí hơn mình sinh lòng ghen ghét làm hại gọi là tật. Như tật đố [嫉妒] ganh ghét. Khuất Nguyên [屈原] : Thế hỗn trọc nhi tật hiền hề, hảo tế mĩ nhi xưng ác [世溷濁而嫉賢兮, 好蔽美而稱惡 (Ly tao [離騷]) Đời hỗn trọc mà ghét người hiền hề, thích che cái tốt mà vạch cái xấu của người.", "meo": "", "reading": "そね.む ねた.む にく.む", "vocab": [("嫉む", "そねむ", "đố"), ("嫉妬", "しっと", "Lòng ghen tị; sự ganh tị")]},
    "挨": {"viet": "AI, ẢI", "meaning_vi": "Đun đẩy.", "meo": "", "reading": "ひら.く", "vocab": [("挨拶", "あいさつ", "lời chào; sự chào hỏi"), ("挨拶する", "あいさつ", "chào; chào hỏi")]},
    "丙": {"viet": "BÍNH", "meaning_vi": "Can Bính, một can trong mười can.", "meo": "", "reading": "ひのえ", "vocab": [("丙", "へい", "Bính (can chi)"), ("丙午", "ひのえうま", "năm Bính Ngọ .")]},
    "窃": {"viet": "THIẾT", "meaning_vi": "Giản thể của chữ 竊", "meo": "", "reading": "ぬす.む ひそ.か", "vocab": [("窃か", "ひそか", "kín đáo"), ("剽窃", "ひょうせつ", "sự ăn cắp")]},
    "拐": {"viet": "QUẢI", "meaning_vi": "bắt cóc, lừa dối", "meo": "Vừa (力) dùng tay (扌) để QUẢI (拐) người đi.", "reading": "kai", "vocab": [("拐帯", "kaidai", "mang theo, mang đi"), ("誘拐", "yuukai", "bắt cóc")]},
    "祉": {"viet": "CHỈ", "meaning_vi": "phúc chỉ, phúc lợi", "meo": "Bộ Thị (示) chỉ Thằng Chó (止) thì sẽ được hưởng phúc Lợi.", "reading": "し", "vocab": [("福祉", "ふくし", "phúc lợi"), ("福利", "ふくり", "phúc lợi, tiền thưởng")]},
    "卸": {"viet": "TÁ", "meaning_vi": "Tháo, cởi. Như hành trang phủ tá [行裝甫卸] vừa trút hành trang xuống.", "meo": "", "reading": "おろ.す おろし おろ.し", "vocab": [("卸", "おろし", "buôn; xỉ"), ("卸し", "おろし", "sự bán buôn")]},
    "寛": {"viet": "KHOAN", "meaning_vi": "Khoan dung", "meo": "", "reading": "くつろ.ぐ ひろ.い ゆる.やか", "vocab": [("寛い", "くつろい", "có tư tưởng rộng rãi"), ("寛ぎ", "くつろぎ", "sự thanh thản")]},
    "硯": {"viet": "NGHIỄN", "meaning_vi": "Cái nghiên mài mực.", "meo": "", "reading": "すずり", "vocab": []},
    "窺": {"viet": "KHUY", "meaning_vi": "Dòm, ngó. Chọc lỗ tường vách để dòm gọi là khuy. Lý Bạch [李白] : Hán há Bạch Đăng đạo, Hồ khuy Thanh Hải loan, Do lai chinh chiến địa, Bất kiến hữu nhân hoàn [漢下白登道，胡窺青海灣，由來征戰地，不見有人還] (Quan san nguyệt [關山月]) Quân Hán đi đường Bạch Đăng, Rợ Hồ dòm ngó vịnh Thanh Hải, Xưa nay nơi chiến địa, Không thấy có người về.", "meo": "", "reading": "うかが.う のぞく", "vocab": [("窺う", "うかがう", "sự đi thăm"), ("窺知", "きち", "sự nhận thức")]},
    "系": {"viet": "HỆ", "meaning_vi": "Buộc, treo. Như hệ niệm [系念] nhớ luôn, để việc vào mỗi nghĩ luôn. Cũng viết là [繫念].", "meo": "", "reading": "ケイ", "vocab": [("系", "けい", "hệ; hệ thống; loại; nhóm; kiểu"), ("体系", "たいけい", "hệ thống; cấu tạo .")]},
    "遜": {"viet": "TỐN", "meaning_vi": "Trốn, lẩn.", "meo": "", "reading": "したが.う へりくだ.る ゆず.る", "vocab": [("不遜", "ふそん", "tính kiêu ngạo"), ("遜色", "そんしょく", "vị trí ở dưới")]},
    "索": {"viet": "TÁC, SÁCH", "meaning_vi": "Dây tơ.", "meo": "", "reading": "サク", "vocab": [("索", "さく", "sợi dây ."), ("索具", "さくぐ", "sự lắp ráp/lắp đặt thiết bị/chằng buộc")]},
    "素": {"viet": "TỐ", "meaning_vi": "Tơ trắng.", "meo": "", "reading": "もと", "vocab": [("素", "もと", "đầu tiên"), ("素人", "しろうと", "người nghiệp dư; người mới vào nghề; người chưa có kinh nghiệm .")]},
    "累": {"viet": "LUY, LŨY, LỤY", "meaning_vi": "Trói.", "meo": "", "reading": "ルイ", "vocab": [("累", "るい", "điều lo lắng"), ("係累", "けいるい", "sự ràng buộc; mối ràng buộc; mối liên hệ; người phụ thuộc")]},
    "螺": {"viet": "LOA", "meaning_vi": "Con ốc. Có thứ ốc trong vỏ nhóng nhánh, thợ sơn hay dùng để dát vào chữ vào đồ cho đẹp gọi là loa điền [螺鈿] khảm ốc.", "meo": "", "reading": "にし にな", "vocab": [("螺子", "ねじ", "vít ."), ("螺旋", "ねじ", "vít; ốc vít; ren")]},
    "繋": {"viet": "HỆ", "meaning_vi": "Kết nối; buộc.", "meo": "", "reading": "つな.ぐ かか.る か.ける", "vocab": [("繋ぐ", "つなぐ", "buộc vào; thắt ."), ("繋争", "けいそう", "cuộc bàn cãi")]},
    "斤": {"viet": "CÂN, CẤN", "meaning_vi": "Cái rìu.", "meo": "", "reading": "キン", "vocab": [("斤", "きん", "kin; ổ"), ("一斤", "いっきん", "một kin")]},
    "芹": {"viet": "CẦN", "meaning_vi": "Rau cần. Thi Kinh [詩經] có câu : Tư nhạo Phán thủy, bạc thái kì cần [思樂泮水, 薄采其芹] Nghĩ thích sông Phán, chút hái rau cần, là bài thơ khen vua Hi Công [僖公] có công sửa lại nhà học phán cung [泮宮]. Vì thế đời sau nói học trò được vào tràng học nhà vua là thái cần [采芹] hay nhập phán [入泮] vậy.", "meo": "", "reading": "せり", "vocab": [("芹", "せり", "cỏ muỗi")]},
    "哲": {"viet": "TRIẾT", "meaning_vi": "Khôn, người hiền trí gọi là triết. Như tiên triết [先哲], tiền triết [前哲] nghĩa là người hiền trí trước.", "meo": "", "reading": "さとい あき.らか し.る さば.く", "vocab": [("中哲", "ちゅうてつ", "Triết học Trung hoa ."), ("哲人", "てつじん", "nhà thông thái; bậc hiền triết; triết gia .")]},
    "逝": {"viet": "THỆ", "meaning_vi": "Đi không trở lại nữa. Vì thế nên gọi người chết là trường thệ [長逝] hay thệ thế [逝世]. Thiền uyển tập anh [禪苑集英] : Kệ tất điệt già nhi thệ [偈畢跌跏而逝] (Khuông Việt Đại sư [匡越大師]) Nói kệ xong, ngồi kiết già mà mất.", "meo": "", "reading": "ゆ.く い.く", "vocab": [("逝く", "ゆく", "chết; qua đời"), ("逝去", "せいきょ", "sự chết; cái chết .")]},
    "誓": {"viet": "THỆ", "meaning_vi": "Răn bảo. Họp các tướng sĩ lại mà răn bảo cho biết kỉ luật gọi là thệ sư [誓師].", "meo": "", "reading": "ちか.う", "vocab": [("誓い", "ちかい", "lời thề"), ("誓う", "ちかう", "thề")]},
    "匠": {"viet": "TƯỢNG", "meaning_vi": "Thợ mộc. Bây giờ thông dụng để gọi cả các thứ thợ. Như đồng tượng [銅匠] thợ đồng, thiết tượng [鐵匠] thợ sắt, v.v.", "meo": "", "reading": "たくみ", "vocab": [("匠", "しょう", "công nhân; người lao động; thợ thủ công; thợ máy; thợ mộc; tiền bạc; giàu có; ý tưởng; ý kiến"), ("名匠", "めいしょう", "nghệ nhân; người thợ có tiếng .")]},
    "析": {"viet": "TÍCH", "meaning_vi": "Gỡ, tẽ ra, chia rẽ. Như li tích [離析] chia ghẽ.", "meo": "", "reading": "セキ", "vocab": [("析出", "せきしゅつ", "sự rút ra"), ("分析", "ぶんせき", "phân tích")]},
    "斥": {"viet": "XÍCH", "meaning_vi": "Bài bác, xua đuổi, vạch rõ", "meo": "Có 'cây' bị 'đinh' (ノ) 'chấm' vào nên bị 'xích' lại.", "reading": "せき", "vocab": [("斥力", "せきりょく", "Lực đẩy"), ("排斥", "はいせき", "Bài xích, loại trừ")]},
    "訴": {"viet": "TỐ", "meaning_vi": "Cáo mách. Như tố oan [訴冤] kêu oan.", "meo": "", "reading": "うった.える", "vocab": [("訴え", "うったえ", "việc kiện cáo; việc tố tụng; đơn kiện; yêu cầu; khiếu nại; kiện tụng; khiếu kiện"), ("上訴", "じょうそ", "chống án")]},
    "楼": {"viet": "LÂU", "meaning_vi": "Giản thể của chữ 樓", "meo": "", "reading": "たかどの", "vocab": [("楼", "ろう", "tháp"), ("妓楼", "ぎろう", "nhà chứa")]},
    "昧": {"viet": "MUỘI", "meaning_vi": "Mờ mờ. Như muội đán [昧旦] mờ mờ sáng.", "meo": "", "reading": "くら.い むさぼ.る", "vocab": [("三昧", "さんまい", "sự lánh mình"), ("愚昧", "ぐまい", "ngu dại")]},
    "魅": {"viet": "MỊ", "meaning_vi": "Si mị [魑魅] loài yêu quái ở gỗ đá hóa ra.", "meo": "", "reading": "ミ", "vocab": [("魅す", "みす", "bỏ bùa mê"), ("魅了", "みりょう", "sự mê hoặc; sự cuốn hút .")]},
    "抹": {"viet": "MẠT", "meaning_vi": "Bôi xoa. Bôi nhằng nhịt gọi là đồ [塗], bôi một vạch thẳng xuống gọi là mạt [抹].", "meo": "", "reading": "マツ", "vocab": [("一抹", "いちまつ", "sự bốc lên"), ("塗抹", "とまつ", "đốm bẩn")]},
    "沫": {"viet": "MẠT", "meaning_vi": "Bọt nổi lên trên mặt nước.", "meo": "", "reading": "あわ しぶき つばき", "vocab": [("泡沫", "ほうまつ", "phù du"), ("飛沫", "しぶき", "bụi nước; giọt nhỏ .")]},
    "朱": {"viet": "CHU", "meaning_vi": "Đỏ.", "meo": "", "reading": "あけ", "vocab": [("朱", "しゅ", "đỏ tươi"), ("丹朱", "たんしゅ", "thủy ngân sulfua")]},
    "殊": {"viet": "THÙ", "meaning_vi": "Dứt, hết tiệt. Như sát nhi vị thù [殺而未殊] giết mà chưa dứt nóc (chém chưa đứt cổ); thù tử [殊死] quyết chết (liều chết); v.v.", "meo": "", "reading": "こと", "vocab": [("殊に", "ことに", "đặc biệt là; một cách đặc biệt; đặc biệt"), ("殊勝", "しゅしょう", "đáng quí; đáng khen; đáng ca ngợi; đáng khâm phục")]},
    "珠": {"viet": "CHÂU", "meaning_vi": "Ngọc châu, tức ngọc trai. Ta thường gọi là trân châu [珍珠].", "meo": "", "reading": "たま", "vocab": [("宝珠", "ほうしゅ", "viên ngọc"), ("念珠", "ねんじゅ", "chuỗi tràng hạt .")]},
    "徘": {"viet": "BỒI", "meaning_vi": "lảng vảng, đi quanh quẩn", "meo": "Hai người đi vòng vòng quanh cây bách.", "reading": "はい", "vocab": [("徘徊", "はいかい", "đi quanh quẩn, lảng vảng; sự đi lang thang"), ("徘徊老人", "はいかいろうじん", "người già hay đi lang thang")]},
    "排": {"viet": "BÀI", "meaning_vi": "Bời ra, gạt ra.", "meo": "", "reading": "ハイ", "vocab": [("排他", "はいた", "sự không cho vào"), ("排便", "はいべん", "sự gạn")]},
    "誹": {"viet": "PHỈ", "meaning_vi": "Chê, thấy người ta làm trái mà mình chê bai gọi là phỉ. Như phỉ báng [誹謗] nói xấu, bêu riếu.", "meo": "", "reading": "そし.る", "vocab": [("誹る", "そしる", "sự vu cáo"), ("誹謗", "ひぼう", "sự phỉ báng .")]},
    "扉": {"viet": "PHI", "meaning_vi": "Cánh cửa. Như đan phi [丹扉] cửa son (cửa nhà vua); sài phi [柴扉] cửa phên (nói cảnh nhà nghèo). Nguyễn Du [阮攸] : Tà nhật yểm song phi [斜日掩窗扉] (Sơn Đường dạ bạc [山塘夜泊]) Mặt trời tà ngậm chiếu cửa sổ.", "meo": "", "reading": "とびら", "vocab": [("扉", "とびら", "cánh cửa ."), ("扉絵", "とびらえ", "tranh đầu sách .")]},
    "斐": {"viet": "PHỈ", "meaning_vi": "đẹp, thanh nhã, tươi sáng", "meo": "Văn (文) trên cùng hình người bị quấn vải phi (非)", "reading": "ヒ", "vocab": [("斐紙", "ひし", "giấy trang trí đẹp"), ("斐然", "ひぜん", "tươi sáng, đẹp đẽ")]},
    "憩": {"viet": "KHẾ", "meaning_vi": "Nghỉ ngơi. Như sảo khế [稍憩] nghỉ một chút.", "meo": "", "reading": "いこ.い いこ.う", "vocab": [("憩", "いこい", "nghỉ ngơi"), ("憩い", "いこい", "nghỉ ngơi")]},
    "臭": {"viet": "XÚ, KHỨU", "meaning_vi": "Mùi. Như kì xú như lan [其臭如蘭] (Dịch Kinh [易經], Hệ Từ thượng [繫辭上]) mùi nó như hoa lan. Bây giờ thì thông dụng để chỉ về mùi hôi thối.", "meo": "", "reading": "くさ.い -くさ.い にお.う にお.い", "vocab": [("臭い", "におい", "mùi; hơi"), ("臭い", "くさい", "hôi thối")]},
    "嗅": {"viet": "KHỨU", "meaning_vi": "Ngửi. Như khứu giác [嗅覺] sự biết, cảm giác về mùi.", "meo": "", "reading": "か.ぐ", "vocab": [("嗅ぐ", "かぐ", "ngửi; hít; hít hà ."), ("嗅覚", "きゅうかく", "khứu giác .")]},
    "鼾": {"viet": "HÃN, HAN", "meaning_vi": "Ngáy. Ngủ ngáy khè khè gọi là hãn [鼾]. $ Có khi đọc là chữ han.", "meo": "", "reading": "いびき", "vocab": [("鼾", "いびき", "sự ngáy; tiếng ngáy"), ("鼾声", "かんせい", "tiếng ngáy")]},
    "喀": {"viet": "KHÁCH", "meaning_vi": "Khách khách [喀喀] tiếng nôn ọe.", "meo": "", "reading": "は.く", "vocab": [("喀痰", "かくたん", "sự khạc"), ("喀血", "かっけつ", "dịch phổi; hộc máu (từ phổi")]},
    "酪": {"viet": "LẠC", "meaning_vi": "Cạo sữa, cách làm dùng nửa gáo sữa cho vào nồi đun qua cho hơi sem sém, rồi lại cho sữa khác vào, đun sôi dần dần mà quấy luôn thấy đặc rồi thì bắc ra, chờ nguội rồi vớt lấy váng mỏng ở trên gọi là tô [酥] còn lại cho một ít dầu sữa cũ vào, lấy giấy mịn kín, thành ra lạc [酪]. Vì thế nên dân miền bắc đều gọi sữa bò sữa ngựa là lạc.", "meo": "", "reading": "ラク", "vocab": [("乳酪", "にゅうらく", "bơ (sữa) ."), ("乾酪", "かんらく", "phó mát")]},
    "賂": {"viet": "LỘ", "meaning_vi": "Đem của đút lót gọi là lộ. Như hối lộ [賄賂].", "meo": "", "reading": "まいな.い まいな.う", "vocab": [("賄賂", "わいろ", "hối lộ"), ("賄賂を使う", "わいろをつかう", "đút lót")]},
    "露": {"viet": "LỘ", "meaning_vi": "Móc, hơi nước gần mặt đất, đêm bám vào cây cỏ, gặp khí lạnh dót lại từng giọt gọi là lộ. Như sương lộ [霜露] sương và móc. Nguyễn Du [阮攸] : Bạch lộ vi sương thu khí thâm [白露爲霜秋氣深] (Thu dạ [秋夜]) Móc trắng thành sương, hơi thu đã già.", "meo": "", "reading": "つゆ", "vocab": [("露", "つゆ", "sương"), ("露わ", "ろわ", "sự phơi")]},
    "妥": {"viet": "THỎA", "meaning_vi": "Yên. Như thỏa thiếp [妥帖] (cũng viết là [妥貼]) đặt yên vừa đúng, thỏa đáng [妥當] vừa khéo vừa đúng, v.v.", "meo": "", "reading": "ダ", "vocab": [("妥協", "だきょう", "sự thỏa hiệp"), ("妥当", "だとう", "hợp lý; đúng đắn; thích đáng")]},
    "如": {"viet": "NHƯ", "meaning_vi": "Bằng, cùng. Dùng để so sánh. Như ái nhân như kỉ [愛人如己] yêu người như yêu mình.", "meo": "", "reading": "ごと.し", "vocab": [("如く", "ごとく", "giống"), ("如し", "ごとし", "ví như; giống như; như là; tương tự")]},
    "茹": {"viet": "NHƯ, NHỰ", "meaning_vi": "Rễ quấn, rễ cây quấn nhau gọi là như. Vì thế quan chức này tiến cử quan chức khác gọi là bạt mao liên như [拔茅連茹].", "meo": "", "reading": "ゆ.でる う.でる", "vocab": [("茹だる", "うだる", "nhọt"), ("茹でる", "ゆでる", "luộc")]},
    "按": {"viet": "ÁN", "meaning_vi": "ấn, đè, xem xét", "meo": "Bàn tay (扌) đặt lên mái nhà (宀) để xem xét, ấn định.", "reading": "あん", "vocab": [("按摩", "あんま", "xoa bóp, mát-xa"), ("按針", "あんじん", "hoa tiêu")]},
    "汝": {"viet": "NHỮ", "meaning_vi": "Sông Nhữ.", "meo": "", "reading": "なんじ なれ い うぬ いまし し しゃ な なむち まし みまし", "vocab": [("爾汝", "じじょ", "anh")]},
    "奴": {"viet": "NÔ", "meaning_vi": "Đứa ở. Luật ngày xưa người nào có tội thì bắt con gái người ấy vào hầu hạ nhà quan gọi là nô tỳ [奴婢], về sau kẻ nào nghèo khó bán mình cho người, mà nương theo về họ người ta cũng gọi là nô.", "meo": "", "reading": "やつ やっこ", "vocab": [("奴", "やつ", "gã ấy; thằng ấy; thằng cha"), ("其奴", "そいつ", "người đó; anh chàng đó; gã đó; người đồng nghiệp đó")]},
    "肝": {"viet": "CAN", "meaning_vi": "Gan, một cơ quan sinh ra nước mật, ở mé tay phải bên bụng, sắc đỏ lờ lờ, có bốn lá.", "meo": "", "reading": "きも", "vocab": [("肝", "きも", "gan"), ("心肝", "しんかん", "tim")]},
    "竿": {"viet": "CAN, CÁN", "meaning_vi": "Cây tre, cần tre, một cành tre gọi là nhất can [一竿]. Ngày xưa viết bằng thẻ tre nên gọi phong thư là can độc [竿牘], lại dùng làm số đo lường (con sào). Như nhật cao tam can [日高三竿] mặt trời cao đã ba ngọn tre, thủy thâm kỷ can [水深幾竿] nước sâu mấy con sào, v.v.", "meo": "", "reading": "さお", "vocab": [("竿", "さお", "cần; trục; cành ."), ("三竿", "さんさお", "sự đi thăm")]},
    "芋": {"viet": "DỤ, HU, VU", "meaning_vi": "Khoai nước. Ta quen đọc là chữ vu.", "meo": "", "reading": "いも", "vocab": [("芋", "いも", "khoai; khoai tây"), ("山芋", "やまいも", "củ từ; khoai mỡ")]},
    "迂": {"viet": "VU", "meaning_vi": "Xa. Con đường không được thẳng suốt gọi là vu. Vì thế nên làm việc không đúng lẽ phải gọi là vu khoát [迂闊] hay vu viễn [迂遠], v.v.", "meo": "", "reading": "ウ", "vocab": [("迂回", "うかい", "khúc ngoặt"), ("迂愚", "うぐ", "ngu dại")]},
    "坪": {"viet": "BÌNH", "meaning_vi": "Chỗ đất bằng phẳng.", "meo": "", "reading": "つぼ", "vocab": [("坪", "つぼ", "tsubo"), ("延坪", "のべつぼ", "tổng diện tích sàn .")]},
    "秤": {"viet": "XỨNG", "meaning_vi": "Cái cân.", "meo": "", "reading": "はかり", "vocab": [("秤", "はかり", "cán cân"), ("天秤", "てんびん", "cái cân đứng")]},
    "匿": {"viet": "NẶC", "meaning_vi": "ẩn nấp, che giấu", "meo": "Chữ THI hài hước trốn trong NHÀ, ngại ngùng NẶC danh.", "reading": "toku", "vocab": [("匿名", "tokumei", "nặc danh"), ("匿う", "kakumau", "che giấu, giấu giếm")]},
    "諾": {"viet": "NẶC", "meaning_vi": "Dạ. Dạ nhanh gọi là dụy [唯], dạ thong thả gọi là nặc [諾]. Thiên nhân nặc nặc, bất như nhất sĩ chi ngạc [千人之諾諾, 不如一士之諤] Nghìn người vâng dạ, không bằng một người nói thẳng.", "meo": "", "reading": "ダク", "vocab": [("一諾", "いちだく", "sự đồng ý"), ("内諾", "ないだく", "sự hứa không chính thức")]},
    "織": {"viet": "CHỨC, CHÍ, XÍ", "meaning_vi": "Dệt, dệt tơ dệt vải đều gọi là chức.", "meo": "", "reading": "お.る お.り おり -おり -お.り", "vocab": [("織", "お", "kiểu"), ("織り", "おり", "kiểu")]},
    "烈": {"viet": "LIỆT", "meaning_vi": "Cháy dữ, lửa mạnh.", "meo": "", "reading": "はげ.しい", "vocab": [("凛烈", "りんれつ", "làm cho đau đớn"), ("劇烈", "げきれつ", "sự dữ dội")]},
    "裂": {"viet": "LIỆT", "meaning_vi": "Xé ra. Sự gì phá hoại gọi là quyết liệt [決裂].", "meo": "", "reading": "さ.く さ.ける -ぎ.れ", "vocab": [("裂く", "さく", "xé; xé rách; xé toạc; chia cắt"), ("亀裂", "きれつ", "cừ")]},
    "葬": {"viet": "TÁNG", "meaning_vi": "Chôn, người chết bỏ vào áo quan đem chôn gọi là táng. Như mai táng [埋葬] chôn cất.", "meo": "", "reading": "ほうむ.る", "vocab": [("葬る", "ほうむる", "chôn cất"), ("仏葬", "ぶっそう", "đám tang nhà Phật .")]},
    "漬": {"viet": "TÍ", "meaning_vi": "Ngâm, tẩm thấm.", "meo": "", "reading": "つ.ける つ.かる -づ.け -づけ", "vocab": [("漬け", "つけ", "dưa chua; dưa góp"), ("塩漬", "しおづけ", "sự muối dưa; sự để cổ phiếu lâu không bán đi trong một thời gian dài vì giá thấp")]},
    "蹟": {"viet": "TÍCH", "meaning_vi": "Cùng nghĩa với chữ tích [跡].", "meo": "", "reading": "あと", "vocab": [("事蹟", "じせき", "tính hiển nhiên; tính rõ ràng"), ("奇蹟", "きあと", "linh dược")]},
    "釘": {"viet": "ĐINH", "meaning_vi": "Cái đinh.", "meo": "", "reading": "くぎ", "vocab": [("釘", "くぎ", "đanh"), ("ねじ釘", "ねじくぎ", "đinh khuy")]},
    "訂": {"viet": "ĐÍNH", "meaning_vi": "Hai bên bàn bạc với nhau cho kĩ rồi mới thỏa thuận gọi là đính. Như đính giao [訂交] đính kết làm bạn, đính ước [訂約].", "meo": "", "reading": "テイ", "vocab": [("修訂", "しゅうてい", "sự sửa"), ("増訂", "ぞうてい", "việc tăng thêm và sửa lại (ấn bản) .")]},
    "寧": {"viet": "NINH, TRỮ", "meaning_vi": "Yên ổn.", "meo": "", "reading": "むし.ろ", "vocab": [("寧", "やすし", "thà... hơn"), ("寧ろ", "むしろ", "thà....còn hơn")]},
    "亭": {"viet": "ĐÌNH", "meaning_vi": "Cái đình, bên đường làm nhà cho khách qua lại trọ gọi là quá nhai đình [過街亭]. Trong các vườn công, xây nhà cho người đến chơi nghỉ ngơi ngắm nghía gọi là lương đình [涼亭].", "meo": "", "reading": "テイ チン", "vocab": [("亭主", "ていしゅ", "ông chủ; người chồng; người chủ nhà"), ("亭亭", "ていてい", "Cao ngất; sừng sững")]},
    "准": {"viet": "CHUẨN, CHUYẾT", "meaning_vi": "Định đúng.", "meo": "", "reading": "ジュン", "vocab": [("准", "じゅん", "chuẩn ."), ("准将", "じゅんしょう", "thiếu tướng hải quân")]},
    "椎": {"viet": "CHUY, CHÙY", "meaning_vi": "Nện, đánh.", "meo": "", "reading": "つち う.つ", "vocab": [("脊椎", "せきつい", "xương sống ."), ("頸椎", "けいつい", "Đốt xương sống cổ tử cung")]},
    "稚": {"viet": "TRĨ", "meaning_vi": "Thơ bé, trẻ bé. Cái gì còn non nớt bé nhỏ đều gọi là trĩ. Con trẻ gọi là trĩ tử [稚子]. Đào Tiềm [陶潛] : Đồng bộc lai nghinh, trĩ tử hậu môn [僮僕來迎, 稚子候門] (Qui khứ lai từ [歸去來辭]) Đầy tớ ra đón, trẻ con đợi ở cửa.", "meo": "", "reading": "いとけない おさない おくて おでる", "vocab": [("丁稚", "でっち", "người học việc"), ("稚児", "ちご", "đứa trẻ; đứa bé")]},
    "推": {"viet": "THÔI, SUY", "meaning_vi": "Đẩy lên.", "meo": "", "reading": "お.す", "vocab": [("推す", "おす", "suy ra; luận ra; kết luận"), ("推力", "すいりょく", "sự đẩy mạnh")]},
    "唯": {"viet": "DUY, DỤY", "meaning_vi": "Độc, chỉ, bui, cũng như chữ duy [惟].", "meo": "", "reading": "ただ", "vocab": [("唯", "ただ", "chỉ; vẻn vẹn chỉ; thế nhưng ."), ("唯々", "いい", "tuyệt đối")]},
    "維": {"viet": "DUY", "meaning_vi": "Buộc. Như duy hệ [維縶] ràng buộc giữ gìn lấy, duy trì [維持] ràng buộc giữ gìn cho khỏi đổ hỏng.", "meo": "", "reading": "イ", "vocab": [("維持", "いじ", "sự duy trì"), ("維新", "いしん", "Duy Tân")]},
    "羅": {"viet": "LA", "meaning_vi": "Cái lưới đánh chim.", "meo": "", "reading": "うすもの", "vocab": [("修羅", "しゅら", "sự chiến đấu"), ("羅典", "らのり", "người La")]},
    "堆": {"viet": "ĐÔI", "meaning_vi": "Đắp.", "meo": "", "reading": "うずたか.い", "vocab": [("堆積", "たいせき", "việc chồng; việc chồng đống (cái gì) ."), ("堆肥", "たいひ", "phân trộn")]},
    "焦": {"viet": "TIÊU, TIỀU", "meaning_vi": "Cháy bỏng, cháy sém.", "meo": "", "reading": "こ.げる こ.がす こ.がれる あせ.る", "vocab": [("焦り", "あせり", "sự thiếu kiên nhẫn"), ("焦る", "あせる", "sốt ruột")]},
    "礁": {"viet": "TIỀU", "meaning_vi": "Đá ngầm, đá mọc ngầm trong nước trong bể, thuyền tầu nhỡ va phải là vỡ.", "meo": "", "reading": "ショウ", "vocab": [("坐礁", "ざしょう", "sự mắc cạn ."), ("岩礁", "がんしょう", "đá ngầm")]},
    "蕉": {"viet": "TIÊU", "meaning_vi": "Gai sống.", "meo": "", "reading": "ショウ", "vocab": []},
    "躍": {"viet": "DƯỢC", "meaning_vi": "Nhảy lên. Mừng nhảy người lên gọi là tước dược [雀躍].", "meo": "", "reading": "おど.る", "vocab": [("躍り", "おどり", "nhấp nhô"), ("躍る", "おどる", "nhảy; nhảy múa")]},
    "擢": {"viet": "TRẠC", "meaning_vi": "Cất lên, nhắc lên. Kẻ đang ở ngôi dưới cất cho lên trên gọi là trạc.", "meo": "", "reading": "ぬ.く ぬき.んでる", "vocab": [("抜擢", "ばってき", "sự lựa chọn"), ("抜擢する", "ばってきする", "đề bạt .")]},
    "奮": {"viet": "PHẤN", "meaning_vi": "Chim dang cánh bay. Chim to sắp bay, tất dang cánh quay quanh mấy cái rồi mới bay lên gọi là phấn.", "meo": "", "reading": "ふる.う", "vocab": [("奮う", "ふるう", "cổ vũ; phấn chấn lên"), ("亢奮", "こうふん", "sự kích thích")]},
    "奪": {"viet": "ĐOẠT", "meaning_vi": "Cướp lấy, lấy hiếp của người ta gọi là đoạt. Như sang đoạt [搶奪] giật cướp, kiếp đoạt [劫奪] cướp bóc, v.v.", "meo": "", "reading": "うば.う", "vocab": [("奪う", "うばう", "cướp"), ("争奪", "そうだつ", "cuộc chiến tranh; trận chiến đấu; sự đấu tranh")]},
    "顧": {"viet": "CỐ", "meaning_vi": "Trông lại, đoái, chỉ về mối tình nhớ nhưng không sao quên được. Như dĩ khứ nhi phục cố [已去而復顧] đã đi mà lại trông lại, cha mẹ yêu con gọi là cố phục [顧復], lời di chiếu của vua gọi là cố mệnh [顧命] cũng là một nghĩa ấy cả. Quên hẳn đi mà không phải vì cố ý gọi là bất cố [不顧] chẳng đoái hoài.", "meo": "", "reading": "かえり.みる", "vocab": [("顧問", "こもん", "sự cố vấn; sự tư vấn; sự khuyên bảo ."), ("回顧", "かいこ", "sự hồi tưởng; sự nhớ lại; hồi tưởng; nhớ lại; nhìn lại; sự nhìn lại")]},
    "雀": {"viet": "TƯỚC", "meaning_vi": "Con chim sẻ.", "meo": "", "reading": "すずめ", "vocab": [("雀", "すずめ", "chim sẻ"), ("孔雀", "くじゃく", "con công trống; con khổng tước")]},
    "隻": {"viet": "CHÍCH, CHỈ", "meaning_vi": "Chiếc, cái gì chỉ có một mình đều gọi là chích. Như hình đan ảnh chích [形單影隻] chiếc bóng đơn hình.", "meo": "", "reading": "セキ", "vocab": [("隻手", "せきしゅ", "một cánh tay; một bàn tay ."), ("数隻", "すうせき", "một vài chiếc (tàu) .")]},
    "穫": {"viet": "HOẠCH", "meaning_vi": "Gặt, cắt lúa.", "meo": "", "reading": "カク", "vocab": [("収穫", "しゅうかく", "thu hoạch (vào mùa gặt) ."), ("収穫する", "しゅうかく", "thu hoạch; gặt hái; gặt về; hái về .")]},
    "獲": {"viet": "HOẠCH", "meaning_vi": "Được, bắt được. Tô Thức [蘇軾] : Thước khởi ư tiền, sử kị trục nhi xạ chi, bất hoạch [鵲起於前, 使騎逐而射之, 不獲] (Phương Sơn Tử truyện [方山子傳]) Chim khách vụt bay trước mặt, sai người cưỡi ngựa đuổi bắn, không được.", "meo": "", "reading": "え.る", "vocab": [("獲る", "える", "lấy được; thu được"), ("乱獲", "らんかく", "vỏ trứng")]},
    "歓": {"viet": "HOAN", "meaning_vi": "Hoan nghênh", "meo": "", "reading": "よろこ.ぶ", "vocab": [("合歓", "ねむ", "Cây bông gòn"), ("歓呼", "かんこ", "Sự tung hô")]},
    "雁": {"viet": "NHẠN", "meaning_vi": "Chim nhạn. Chim nhạn, mùa thu lại, mùa xuân đi, cho nên gọi là hậu điểu [候鳥] chim mùa. Chim nhạn bay có thứ tự, nên anh em gọi là nhạn tự [雁序]. Có khi viết là nhạn [鴈]. Ta gọi là con chim mòng. Nguyễn Trãi [阮廌] : Cố quốc tâm quy lạc nhạn biên [故國心歸落雁邊] (Thần Phù hải khẩu [神苻海口]) Lòng mong về quê cũ theo cánh nhạn sa.", "meo": "", "reading": "かり かりがね", "vocab": [("雁書", "がんしょ", "chữ cái"), ("雁木", "がんぎ", "lối thoát ra")]},
    "擁": {"viet": "ỦNG, UNG", "meaning_vi": "Ôm, cầm. Vương An Thạch [王安石] : Dư dữ tứ nhân ủng hỏa dĩ nhập [余與四人擁火以入] (Du Bao Thiền Sơn kí [遊褒禪山記]) Tôi cùng bốn người cầm đuốc đi vô (hang núi).", "meo": "", "reading": "ヨウ", "vocab": [("擁する", "ようする", "có"), ("抱擁", "ほうよう", "sự ôm chặt .")]},
    "雄": {"viet": "HÙNG", "meaning_vi": "Con đực. Các loài có lông thuộc về giống đực gọi là hùng. Giống thú đực cũng gọi là hùng.", "meo": "", "reading": "お- おす おん", "vocab": [("雄", "おす", "đực ."), ("両雄", "りょうゆう", "hai vỹ nhân; hai anh hùng .")]},
    "裸": {"viet": "LỎA, KHỎA", "meaning_vi": "Trần truồng. Ta quen đọc khỏa.", "meo": "", "reading": "はだか", "vocab": [("裸", "はだか", "sự trần trụi; sự trơ trụi; sự lõa thể"), ("丸裸", "まるはだか", "trần")]},
    "彙": {"viet": "VỊ, VỰNG, HỐI", "meaning_vi": "Loài, xếp từng loại với nhau gọi là vị tập [彙集]. Ta quen đọc là vựng.", "meo": "", "reading": "はりねずみ", "vocab": [("彙報", "いほう", "thông cáo"), ("語彙", "ごい", "từ vựng; ngôn từ")]},
    "弾": {"viet": "ĐÀN, ĐẠN", "meaning_vi": "Đánh đàn; viên đạn", "meo": "", "reading": "ひ.く -ひ.き はず.む たま はじ.く はじ.ける ただ.す はじ.きゆみ", "vocab": [("弾", "たま", "viên đạn"), ("弾く", "ひく", "chơi (nhạc cụ)")]},
    "蝉": {"viet": "THIỀN", "meaning_vi": "Giản thể của chữ [蟬].", "meo": "", "reading": "せみ", "vocab": [("蝉", "せみ", "ve sầu ."), ("川蝉", "かわせみ", "Chim bói cá .")]},
    "禅": {"viet": "THIỆN, THIỀN", "meaning_vi": "Tục dùng như chữ thiền [禪].", "meo": "", "reading": "しずか ゆず.る", "vocab": [("禅", "ぜん", "phái thiền"), ("禅僧", "ぜんそう", "nhà sư theo phái thiền; thiền tăng .")]},
    "箪": {"viet": "ĐAN", "meaning_vi": "Giản thể của chữ 簞", "meo": "", "reading": "はこ", "vocab": [("瓢箪", "ひょうたん", "bầu; bí ."), ("箪笥", "たんす", "tủ")]},
    "淫": {"viet": "DÂM", "meaning_vi": "Quá, phàm cái gì quá lắm đều gọi là dâm. Như dâm vũ [淫雨] mưa dầm, dâm hình [淫刑] hình phạt ác quá, v.v. Nguyễn Du [阮攸] : Dâm thư do thắng vị hoa mang [淫書猶勝爲花忙] (Điệp tử thư trung [蝶死書中]) Say đắm vào sách còn hơn đa mang vì hoa.", "meo": "", "reading": "ひた.す ほしいまま みだ.ら みだ.れる みだり", "vocab": [("淫", "いん", "dấu ."), ("淫ぷ", "いんぷ", "dâm phụ .")]},
    "廷": {"viet": "ĐÌNH", "meaning_vi": "Triều đình, chỗ phát chánh lệnh cho cả nước theo gọi là đình. Như đình đối [廷對] vào đối đáp ở chốn triều đình, đình nghị [廷議] sự bàn ở trong triều đình.", "meo": "", "reading": "テイ", "vocab": [("廷丁", "ていてい", "cao"), ("入廷", "にゅうてい", "sự vào phòng xử án; sự vào pháp đình (thẩm phán")]},
    "艇": {"viet": "ĐĨNH", "meaning_vi": "Cái thoi, thứ thuyền nhỏ mà dài. Nguyễn Du [阮攸] : Khẩn thúc giáp điệp quần, Thái liên trạo tiểu đĩnh [緊束蛺蝶裙, 採蓮棹小艇] (Mộng đắc thái liên [夢得埰蓮]) Buộc chặt quần cánh bướm, Hái sen chèo thuyền con.", "meo": "", "reading": "テイ", "vocab": [("艇庫", "ていこ", "kho đặt trên tàu"), ("漕艇", "そうてい", "sự liên kết")]},
    "挺": {"viet": "ĐĨNH", "meaning_vi": "Trội cao. Như thiên đĩnh chi tư [天挺之資] tư chất trời sinh trội hơn cả các bực thường.", "meo": "", "reading": "ぬ.く", "vocab": [("空挺", "くうてい", "không vận"), ("挺身", "ていしん", "quân tình nguyện")]},
    "旋": {"viet": "TOÀN", "meaning_vi": "Trở lại. Như khải toàn [凱旋] thắng trận trở về.", "meo": "", "reading": "セン", "vocab": [("凱旋", "がいせん", "sự khải hoàn; chiến thắng trở về; khải hoàn"), ("周旋", "しゅうせん", "sự chuyền nhau; sự luân chuyển; luân chuyển .")]},
    "鯨": {"viet": "KÌNH", "meaning_vi": "Cá kình (voi). Hình tuy giống cá mà thực ra thuộc về loài thú. Có con to dài đến tám chín mươi thước. Nguyễn Trãi [阮廌] : Ngao phụ xuất sơn sơn hữu động, Kình du tắc hải hải vi trì [鰲負出山山有洞, 鯨遊塞海海爲池] (Long Đại Nham [龍袋岩]) Con ba ba đội núi nổi lên, núi có động, Cá kình bơi lấp biển, biển thành ao.", "meo": "", "reading": "くじら", "vocab": [("鯨", "くじら", "cá voi"), ("鯨座", "くじらざ", "cá voi")]},
    "掠": {"viet": "LƯỢC", "meaning_vi": "Cướp lấy. Như xâm lược [侵掠] lấn tới mà cướp bóc.", "meo": "", "reading": "かす.める かす.る かす.れる", "vocab": [("掠り", "かすり", "sự thả súc vật cho ăn cỏ"), ("掠る", "かする", "chỗ da bị sầy")]},
    "憬": {"viet": "CẢNH", "meaning_vi": "Hiểu biết, tỉnh ngộ, có ý lo sợ mà tỉnh cơn mê ra gọi là cảnh nhiên [憬然].", "meo": "", "reading": "あこが.れる", "vocab": [("憧憬", "どうけい", "ước ao")]},
    "撲": {"viet": "PHÁC, BẠC, PHỐC", "meaning_vi": "Đánh, dập tắt.", "meo": "", "reading": "ボク", "vocab": [("打撲", "だぼく", "vết thâm tím"), ("撲滅", "ぼくめつ", "sự tiêu diệt; sự hủy diệt; sự triệt tiêu")]},
    "叢": {"viet": "TÙNG", "meaning_vi": "Hợp. Sưu tập số nhiều để vào một chỗ gọi là tùng. Như tùng thư [叢書], tùng báo [叢報] tích góp nhiều sách báo làm một bộ, một loại.", "meo": "", "reading": "くさむら むら.がる むら", "vocab": [("叢", "くさむら", "bụi cây"), ("一叢", "いちくさむら", "coppice")]},
    "尤": {"viet": "VƯU", "meaning_vi": "Lạ, rất, càng. Như thù vưu tuyệt tích [殊尤絕迹] lạ lùng hết mực, nghĩa là nó khác hẳn các cái tầm thường. Con gái đẹp gọi là vưu vật [尤物].", "meo": "", "reading": "もっと.も とが.める", "vocab": [("尤も", "もっとも", "khá đúng; có lý"), ("尤度", "ゆうど", "sự có vẻ hợp lý")]},
    "蹴": {"viet": "XÚC", "meaning_vi": "Bước xéo gót, rảo bước theo sau gọi là xúc.", "meo": "", "reading": "け.る", "vocab": [("蹴る", "ける", "đá"), ("一蹴", "いっしゅう", "trôn chai")]},
    "枕": {"viet": "CHẨM, CHẤM", "meaning_vi": "Xương trong óc cá.", "meo": "", "reading": "まくら", "vocab": [("枕", "まくら", "cái gối"), ("仮枕", "かりまくら", "giấc ngủ chợp")]},
    "耽": {"viet": "ĐAM", "meaning_vi": "Vui, quá vui gọi là đam.", "meo": "", "reading": "ふ.ける", "vocab": [("耽る", "ふける", "buông thả mình"), ("耽溺", "たんでき", "sự nuông chiều")]},
    "桟": {"viet": "SẠN", "meaning_vi": "Giá để đồ", "meo": "", "reading": "かけはし", "vocab": [("桟敷", "さじき", "hộp"), ("桟橋", "さんきょう", "bến tàu; bến")]},
    "銭": {"viet": "TIỀN", "meaning_vi": "Tiền bạc", "meo": "", "reading": "ぜに すき", "vocab": [("銭", "せん", "một phần trăm của một yên; một hào ."), ("借銭", "しゃくせん", "nợ")]},
    "践": {"viet": "TIỄN", "meaning_vi": "thực tiễn, thực hành", "meo": "Kim (金) loại đạp (践) lên cỏ (艹) để thực tiễn.", "reading": "せん", "vocab": [("実践", "じっせん", "thực tiễn, thực hành"), ("踏み践る", "ふみにじる", "chà đạp, giày xéo")]},
    "肖": {"viet": "TIẾU, TIÊU", "meaning_vi": "Giống. Nguyễn Du [阮攸] : Kim chi họa đồ vô lược tiếu[今之畫圖無略肖] (Mạnh Tử từ cổ liễu [孟子祠古柳]) Những bức vẽ ngày nay không giống chút nào.", "meo": "", "reading": "あやか.る", "vocab": [("不肖", "ふしょう", "sự thiếu khả năng; sự thiếu trình độ"), ("肖像", "しょうぞう", "chân dung")]},
    "宵": {"viet": "TIÊU", "meaning_vi": "Đêm. Như trung tiêu [中宵] nửa đêm. Nguyễn Du [阮攸] có bài thơ nhan đề là Quỳnh Hải nguyên tiêu [瓊海元宵] Đêm rằm tháng giêng ở Quỳnh Hải.", "meo": "", "reading": "よい", "vocab": [("宵", "よい", "chiều; chiều muộn"), ("今宵", "こよい", "night) /tə'nait/")]},
    "硝": {"viet": "TIÊU", "meaning_vi": "Đá tiêu. Chất trong suốt, đốt cháy dữ, dùng làm thuốc súng thuốc pháo và nấu thủy tinh.", "meo": "", "reading": "ショウ", "vocab": [("硝化", "しょうか", "sự nitrat hoá"), ("硝子", "がらす", "kính thuỷ tinh")]},
    "屑": {"viet": "TIẾT", "meaning_vi": "Mạt vụn. Như thiết tiết [鐵屑] mạt sắt.", "meo": "", "reading": "くず いさぎよ.い", "vocab": [("屑", "くず", "rác"), ("屑屋", "くずや", "người buôn bán giẻ rách; người bán hàng đồng nát")]},
    "削": {"viet": "TƯỚC", "meaning_vi": "Vót.", "meo": "", "reading": "けず.る はつ.る そ.ぐ", "vocab": [("削", "けず", "plane"), ("削ぐ", "そぐ", "vỏ bào")]},
    "鎖": {"viet": "TỎA", "meaning_vi": "Cái vòng. Lấy vòng móc liền nhau gọi là tỏa. Vì thế những vòng xúc xích đeo làm đồ trang sức gọi là liên tỏa [連鎖], lấy vòng móc liền nhau làm áo dày gọi là tỏa tử giáp [鎖子甲].", "meo": "", "reading": "くさり とざ.す", "vocab": [("鎖", "くさり", "cái xích; xích"), ("鎖国", "さこく", "bế quan tỏa cảng .")]},
    "貞": {"viet": "TRINH", "meaning_vi": "Trinh, chính đính, giữ được tấm lòng chính đính thủy chung không ai làm lay động được gọi là trinh. Như trung trinh [忠貞] trong sạch không thay lòng, kiên trinh [堅貞] trong sạch vững bền, v.v... Đàn bà không thất tiết (không yêu ai, chỉ yêu một chồng) gọi là trinh phụ [貞婦]. Con gái chính đính (không theo trai) gọi là trinh nữ [貞女].", "meo": "", "reading": "さだ", "vocab": [("不貞", "ふてい", "không trung thành; không chung thủy"), ("貞女", "ていじょ", "Phụ nữ tiết hạnh; vợ chung thủy")]},
    "偵": {"viet": "TRINH", "meaning_vi": "Rình xét. Như trinh thám [偵探] dò xét.", "meo": "", "reading": "テイ", "vocab": [("偵", "てい", "gián điệp ."), ("内偵", "ないてい", "việc điều tra bí mật")]},
    "貫": {"viet": "QUÁN", "meaning_vi": "Cái dây xâu tiền, cho nên gọi một xâu nghìn đồng tiền là nhất quán [一貫] (một quan). Như vạn quán gia tư [萬貫家私] nhà giàu có đến vạn quan. Tính số tham tàng trộm cắp, tích chứa được đủ số bao nhiêu đó gọi là mãn quán [滿貫] nghĩa là như xâu tiền đã đủ quan, cho đến hết cữ vậy, vì thế nên tội ác đến cùng cực gọi là ác quán mãn doanh [惡貫滿盈].", "meo": "", "reading": "つらぬ.く ぬ.く ぬき", "vocab": [("貫く", "つらぬく", "xuyên qua; xuyên thủng qua; xuyên suốt; quán triệt ."), ("一貫", "いっかん", "nhất quán")]},
    "惧": {"viet": "CỤ", "meaning_vi": "Như chữ cụ [懼].", "meo": "", "reading": "おそ.れる", "vocab": [("危惧", "きく", "sự sợ hãi; sự lo âu; nỗi lo; lo lắng; lo"), ("危惧", "きぐ", "sự sợ sệt; sự sợ hãi; sợ sệt; sợ hãi")]},
    "慎": {"viet": "THẬN", "meaning_vi": "Ghín, cẩn thận. Đỗ Phủ [杜甫] : Thận vật xuất khẩu tha nhân thư [慎勿出口他人狙] (Ai vương tôn [哀王孫]) Cẩn thận giữ miệng, (coi chừng) kẻ khác rình dò.", "meo": "", "reading": "つつし.む つつし つつし.み", "vocab": [("慎み", "つつしみ", "tính khiêm tốn"), ("慎む", "つつしむ", "cẩn thận; thận trọng; nín nhịn")]},
    "鎮": {"viet": "TRẤN", "meaning_vi": "Một dạng của chữ trấn [鎭].", "meo": "", "reading": "しず.める しず.まる おさえ", "vocab": [("鎮圧", "ちんあつ", "sự trấn áp"), ("鎮守", "ちんじゅ", "sự phái binh sĩ đến trấn thủ; thần thổ công; thổ địa")]},
    "填": {"viet": "ĐIỀN, TRẦN, ĐIỄN, TRẤN", "meaning_vi": "Lấp, lấp cho đầy hố gọi là điền. Lý Hoa [李華] : Thi điền cự cảng chi ngạn, huyết mãn trường thành chi quật [屍填巨港之岸, 血滿長城之窟] (Điếu cổ chiến trường văn [弔古戰場文]) Thây lấp bờ cảng lớn, máu ngập đầy hào trường thành.", "meo": "", "reading": "は.まる は.める うず.める しず.める ふさ.ぐ", "vocab": [("填る", "はまる", "(từ cổ"), ("充填", "じゅうてん", "nút (chậu sứ rửa mặt")]},
    "惜": {"viet": "TÍCH", "meaning_vi": "Tiếc, quý trọng", "meo": "Tim (忄) có người (昔) nhớ tiếc những điều đã qua.", "reading": "o.shii", "vocab": [("惜しむ", "o.shimu", "tiếc nuối"), ("惜別", "sekibetsu", "chia tay lưu luyến")]},
    "錯": {"viet": "THÁC, THỐ", "meaning_vi": "Hòn đá ráp, đá mài. Thi Kinh [詩經] có câu tha sơn chi thạch, khả dĩ vi thác [他山之石可以爲錯] đá ở núi khác có thể lấy làm đá mài. Ý nói bè bạn hay khuyên ngăn cứu chính lại lỗi lầm cho mình.", "meo": "", "reading": "サク シャク", "vocab": [("錯乱", "さくらん", "sự lộn xôn"), ("交錯", "こうさく", "hỗn hợp; lẫn lộn; sự trộn lẫn với nhau; sự pha lẫn vào nhau; pha trộn")]},
    "措": {"viet": "THỐ, TRÁCH", "meaning_vi": "Thi thố ra.", "meo": "", "reading": "お.く", "vocab": [("措く", "おく", "trừ ra"), ("措定", "そてい", "sự mang")]},
    "跳": {"viet": "KHIÊU", "meaning_vi": "Nhảy. Như khiêu vũ xướng ca [跳舞唱歌] nhảy múa ca hát.", "meo": "", "reading": "は.ねる と.ぶ -と.び", "vocab": [("跳ぶ", "とぶ", "nhảy lên; bật lên; nhảy"), ("跳ねる", "はねる", "bắn")]},
    "挑": {"viet": "THIÊU, THIỂU, THAO, KHIÊU", "meaning_vi": "Gánh. Tô Mạn Thù [蘇曼殊] : Ngô nhật gian thiêu hoa dĩ thụ phú nhân [吾日間挑花以售富人] (Đoạn hồng linh nhạn kí [斷鴻零雁記]) Ban ngày cháu gánh hoa đem bán cho nhà giàu có.", "meo": "", "reading": "いど.む", "vocab": [("挑む", "いどむ", "thách thức"), ("挑戦", "ちょうせん", "sự thách thức")]},
    "眺": {"viet": "THIẾU", "meaning_vi": "Ngắm xa. Tô Mạn Thù [蘇曼殊] : Xuất sơn môn thiếu vọng [出山門眺望] (Đoạn hồng linh nhạn kí [斷鴻零雁記]) Bước ra cổng chùa ngắm ra xa.", "meo": "", "reading": "なが.める", "vocab": [("眺め", "ながめ", "tầm nhìn"), ("眺める", "ながめる", "nhìn; ngắm")]},
    "桃": {"viet": "ĐÀO", "meaning_vi": "quả đào", "meo": "Cây (木) trồng trên gò đất (土) thì có trái đào mọng nước (兆).", "reading": "もも", "vocab": [("桃", "もも", "quả đào"), ("白桃", "はくとう", "đào trắng")]},
    "掲": {"viet": "YẾT", "meaning_vi": "Yết thị", "meo": "", "reading": "かか.げる", "vocab": [("前掲", "ぜんけい", "đã nói ở trên"), ("掲げる", "かかげる", "treo")]},
    "謁": {"viet": "YẾT", "meaning_vi": "Yết kiến, vào hầu chuyện, người hèn mọn muốn xin vào hầu bực tôn quý, để bẩm bạch sự gì gọi là yết.", "meo": "", "reading": "エツ", "vocab": [("内謁", "ないえつ", "Cuộc gặp mặt không chính thức với người cấp trên ."), ("謁する", "えっする", "xem; thưởng thức")]},
    "喝": {"viet": "HÁT, ỚI", "meaning_vi": "Quát mắng. Như lệ thanh hát đạo [厲聲喝道] quát lớn tiếng.", "meo": "", "reading": "カツ", "vocab": [("一喝", "いっかつ", "tiếng gầm"), ("威喝", "いかつ", "sự đe doạ")]},
    "靄": {"viet": "ÁI", "meaning_vi": "Khí mây. Như yên ái [煙靄] khí mây mù như khói. Trần Nhân Tông [陳仁宗] : Cổ tự thê lương thu ái ngoại [古寺淒涼秋靄外] (Lạng Châu vãn cảnh [諒州晚景]) Chùa cổ lạnh lẽo trong khí mây mùa thu.", "meo": "", "reading": "もや", "vocab": [("靄", "もや", "sương mù ."), ("朝靄", "あさもや", "sương mù buổi sáng")]},
    "褐": {"viet": "HẠT, CÁT", "meaning_vi": "Áo vải to. Như đoản hạt [短褐] quần áo ngắn vải thô.", "meo": "", "reading": "カツ", "vocab": [("褐炭", "かったん", "than bùn"), ("褐色", "かっしょく", "màu nâu")]},
    "葛": {"viet": "CÁT", "meaning_vi": "Dây sắn. Rễ dùng làm thuốc gọi là cát căn [葛根], vỏ dùng dệt vải gọi là cát bố [葛布].", "meo": "", "reading": "つづら くず", "vocab": [("葛藤", "かっとう", "sự xung đột"), ("葛折り", "かずらおり", "khúc lượn")]},
    "鞭": {"viet": "TIÊN", "meaning_vi": "Hình roi. Một thứ dùng trong hình pháp để đánh người.", "meo": "", "reading": "むち むちうつ", "vocab": [("先鞭", "せんべん", "bắt đầu"), ("鞭撻", "べんたつ", "sự làm can đảm")]},
    "梗": {"viet": "NGẠNH", "meaning_vi": "Cành cây.", "meo": "", "reading": "ふさぐ やまにれ おおむね", "vocab": [("梗塞", "こうそく", "sự nhồi máu; sự nghẽn mạch; nhồi máu; nghẽn mạch ."), ("桔梗", "ききょう", "chứng tràn khí ngực")]},
    "胞": {"viet": "BÀO", "meaning_vi": "Bào thai [胞胎], lúc con còn ở trong bụng mẹ, ngoài có cái mạng bao bọc gọi là bào (nhau).", "meo": "", "reading": "ホウ", "vocab": [("同胞", "どうほう", "đồng bào; người cùng một nước ."), ("胞子", "ほうし", "bào tử [thực vật]")]},
    "飽": {"viet": "BÃO", "meaning_vi": "No, ăn no. Nguyễn Du [阮攸] : Chỉ đạo Trung Hoa tẫn ôn bão, Trung Hoa diệc hữu như thử nhân [只道中華盡溫飽, 中華亦有如此人] (Thái Bình mại ca giả [太平賣歌者]) Chỉ nghe nói ở Trung Hoa đều được no ấm, Thế mà Trung Hoa cũng có người (đói khổ) như vậy sao ?", "meo": "", "reading": "あ.きる あ.かす あ.く", "vocab": [("飽き", "あき", "sự mệt mỏi; sự chán nản"), ("飽和", "ほうわ", "sự bão hòa")]},
    "鞄": {"viet": "BẠC, BÀO", "meaning_vi": "Thợ thuộc da.", "meo": "", "reading": "かばん", "vocab": [("鞄", "かばん", "cặp; túi; balô; cặp sách; túi xách; giỏ"), ("折り鞄", "おりかばん", "cái cặp để giấy tờ")]},
    "哺": {"viet": "BỘ", "meaning_vi": "Mớm (chim mẹ mớm cho chim con).", "meo": "", "reading": "はぐく.む ふく.む", "vocab": [("哺乳", "ほにゅう", "sự sinh sữa"), ("哺育", "ほいく", "sự chăm sóc bệnh nhân")]},
    "浦": {"viet": "PHỔ, PHỐ", "meaning_vi": "Bến sông, ngạch sông đổ ra bể. Nguyễn Du [阮攸] : Hồi thủ Lam Giang phố [回首藍江浦] (Thu chí [秋至]) Ngoảnh đầu về bến sông Lam.", "meo": "", "reading": "うら", "vocab": [("浦", "うら", "cái vịnh nhỏ; vịnh nhỏ"), ("浦波", "うらなみ", "sóng bên bờ biển; sóng biển gần bờ")]},
    "舗": {"viet": "PHỐ", "meaning_vi": "lát, đường phố", "meo": "Có THỔ (土) trên MÁI NHÀ (甫) để lát PHỐ.", "reading": "ほ", "vocab": [("舗装", "ほそう", "sự lát (đường), vỉa hè"), ("店舗", "てんぽ", "cửa hàng")]},
    "奇": {"viet": "KÌ, CƠ", "meaning_vi": "Lạ. Vật hiếm có mà khó kiếm gọi là kì. Cao Bá Quát [高伯适] : Phong cảnh dĩ kì tuyệt [風景已奇絕] (Quá Dục Thúy sơn [過浴翠山]) Phong cảnh thật đẹp lạ.", "meo": "", "reading": "く.しき あや.しい くし めずら.しい", "vocab": [("伝奇", "でんき", "truyền kỳ (truyện)"), ("奇体", "きたい", "lạ")]},
    "騎": {"viet": "KỊ", "meaning_vi": "Cưỡi ngựa.", "meo": "", "reading": "キ", "vocab": [("騎乗", "きじょう", "núi"), ("騎兵", "きへい", "kị binh; kỵ binh")]},
    "綺": {"viet": "KHỈ, Ỷ", "meaning_vi": "Các thứ the lụa có hoa bóng chằng chịt không dùng sợi thẳng, đều gọi là khỉ.", "meo": "", "reading": "あや", "vocab": [("綺麗", "きれい", "xinh"), ("綺想曲", "あやぎぬそうきょく", "khúc tuỳ hứng")]},
    "椅": {"viet": "Y, Ỷ", "meaning_vi": "Cây y, một loài cây lớn, lá hình trái tim, mùa hạ nở hoa màu vàng, gỗ dùng được. Còn có tên là sơn đồng tử [山桐子].", "meo": "", "reading": "イ", "vocab": [("椅子", "いす", "ghế; cái ghế"), ("寝椅子", "ねいす", "Ghế dài; đi văng; trường kỷ .")]},
    "儀": {"viet": "NGHI", "meaning_vi": "Dáng. Như uy nghi [威儀] có cái dáng nghiêm trang đáng sợ.", "meo": "", "reading": "ギ", "vocab": [("儀", "ぎ", "phép tắc"), ("一儀", "いちぎ", "vốn có")]},
    "犠": {"viet": "HI SINH", "meaning_vi": "hi sinh, hy sinh", "meo": "Con BÒ (牛) bị Ném Đá (殳) trên Mâm Cỗ (米) để HI SINH.", "reading": "ぎ", "vocab": [("犠牲", "ぎせい", "hi sinh, hy sinh"), ("犠牲者", "ぎせいしゃ", "nạn nhân, người hi sinh")]},
    "蟻": {"viet": "NGHĨ", "meaning_vi": "Con kiến.", "meo": "", "reading": "あり", "vocab": [("蟻", "あり", "con kiến"), ("蟻塚", "ありづか", "ụ kiến .")]},
    "也": {"viet": "DÃ", "meaning_vi": "Vậy, nhời nói hết câu. Như nghĩa giả nghi dã [義者宜也] nghĩa, ấy là sự nên thế thì làm vậy.", "meo": "", "reading": "なり か また", "vocab": [("可也", "かなり", "kha khá; đáng chú ý; khá")]},
    "施": {"viet": "THI, THÍ, DỊ, THỈ", "meaning_vi": "Bày ra, đặt ra, đem dùng ra cho người hay vật gọi là thi. Như thi thuật [施術] làm thuật cho kẻ nào, thi trị [施治] làm phép chữa cho kẻ nào, thi ân [施恩] ra ơn cho kẻ nào, thi phấn [施粉] đánh phấn, thi lễ [施禮] làm lễ chào, v.v.", "meo": "", "reading": "ほどこ.す", "vocab": [("施し", "ほどこし", "lòng nhân đức"), ("施す", "ほどこす", "bố thí")]},
    "弛": {"viet": "THỈ", "meaning_vi": "Buông dây cung.", "meo": "", "reading": "たる.む たる.める たゆ.む ゆる.む ゆる.み", "vocab": [("弛み", "ゆるみ", "uể oải"), ("弛む", "たるむ", "lơi lỏng")]},
    "嗣": {"viet": "TỰ", "meaning_vi": "Nối. Như tự tử [嗣子] con nối.", "meo": "", "reading": "シ", "vocab": [("嫡嗣", "ちゃくし", "đích tự; người thừa kế hợp pháp ."), ("嗣子", "しし", "người thừa kế; người thừa tự .")]},
    "覗": {"viet": "THẤU", "meaning_vi": "nhìn trộm, liếc", "meo": "Mắt (目) của người thợ (工) xây nhà ở phía tây (西) hay nhìn trộm.", "reading": "のぞく", "vocab": [("覗き", "のぞき", "sự nhìn trộm"), ("覗き込む", "のぞきこむ", "nhìn vào bên trong")]},
    "笥": {"viet": "TỨ", "meaning_vi": "Cái sọt vuông, thùng vuông.", "meo": "", "reading": "け はこ", "vocab": [("箪笥", "たんす", "tủ"), ("用箪笥", "ようだんす", "tủ com")]},
    "嫡": {"viet": "ĐÍCH", "meaning_vi": "Vợ cả, con vợ cả gọi là đích tử [嫡子].", "meo": "", "reading": "チャク テキ", "vocab": [("嫡嗣", "ちゃくし", "đích tự; người thừa kế hợp pháp ."), ("嫡子", "ちゃくし", "đích tử; con hợp pháp")]},
    "摘": {"viet": "TRÍCH", "meaning_vi": "Hái. Như trích qua [摘瓜] hái dưa, trích quả [摘果] hái quả, v.v.", "meo": "", "reading": "つ.む", "vocab": [("摘む", "つまむ", "nắm; nhặt (bằng đầu ngón tay)"), ("摘む", "つむ", "hái")]},
    "釣": {"viet": "ĐIẾU", "meaning_vi": "Câu cá. Nguyễn Trãi [阮廌] : Bản thị canh nhàn điếu tịch nhân [本是耕閒釣寂人] (Đề Từ Trọng Phủ canh ẩn đường [題徐仲甫耕隱堂]) Ta vốn là kẻ cày nhàn, câu tịch.", "meo": "", "reading": "つ.る つ.り つ.り-", "vocab": [("釣", "つり", "sự đánh cá"), ("お釣", "おつり", "tiền thối lại .")]},
    "灼": {"viet": "CHƯỚC", "meaning_vi": "Đốt, nướng.", "meo": "", "reading": "あらた やく", "vocab": [("灼ける", "やける", "(Ê"), ("灼熱", "しゃくねつ", "sự nóng sáng")]},
    "棟": {"viet": "ĐỐNG", "meaning_vi": "Nóc mái nhà. Vật gì nhiều lắm gọi là hãn ngưu sung đống [汗牛充棟] mồ hôi trâu rui trên nóc.", "meo": "", "reading": "むね むな-", "vocab": [("棟", "とう", "khu vực; tòa nhà"), ("棟", "むね", "nóc nhà")]},
    "煉": {"viet": "LUYỆN", "meaning_vi": "Nung đúc, rèn đúc, xem chữ luyện [鍊].", "meo": "", "reading": "ね.る", "vocab": [("煉乳", "れんにゅう", "sữa đặc"), ("修煉", "おさむねり", "sự mở mang")]},
    "錬": {"viet": "LUYỆN", "meaning_vi": "Tinh luyện, rèn luyện", "meo": "", "reading": "ね.る", "vocab": [("修錬", "しゅうれん", "sự mở mang"), ("錬成", "れんせい", "sự huấn luyện; sự đào tạo")]},
    "陳": {"viet": "TRẦN, TRẬN", "meaning_vi": "Bày. Như trần thiết [陳設] bày đặt.", "meo": "", "reading": "ひ.ねる", "vocab": [("陳列", "ちんれつ", "sự trưng bày"), ("陳弁", "ちんべん", "sự phân trần .")]},
    "渉": {"viet": "THIỆP", "meaning_vi": "Can thiệp, giao thiệp", "meo": "", "reading": "わた.る", "vocab": [("渉る", "わたるる", "đi qua"), ("交渉", "こうしょう", "sự đàm phán; cuộc đàm phán; đàm phán")]},
    "頻": {"viet": "TẦN", "meaning_vi": "Luôn. Phần nhiều dùng làm trợ từ. Như tần tần [頻頻] luôn luôn. Nguyễn Du [阮攸] : Chinh mã tần tần kinh thất lộ [征馬頻頻驚失路] (Dự Nhượng kiều chủy thủ hành [豫讓橋匕首行]) Ngựa chiến nhiều lần hí lên sợ lạc đường.", "meo": "", "reading": "しき.りに", "vocab": [("頻々", "ひんぴん", "sự tấp nập; sự nhiều lần"), ("頻出", "ひんしゅつ", "chung")]},
    "瀕": {"viet": "TẦN", "meaning_vi": "Bến.", "meo": "", "reading": "ほとり", "vocab": [("瀕死", "ひんし", "sự chết")]},
    "捗": {"viet": "DUỆ", "meaning_vi": "Cũng như chữ duệ [曳].", "meo": "", "reading": "はかど.る", "vocab": [("捗る", "はかどる", "tiến bộ"), ("進捗", "しんちょく", "sự tiến tới")]},
    "幹": {"viet": "CÁN, CAN", "meaning_vi": "Mình. Như khu cán [軀幹] vóc người, mình người.", "meo": "", "reading": "みき", "vocab": [("幹", "かん", "thân cây"), ("幹", "みき", "thân cây .")]},
    "斡": {"viet": "OÁT", "meaning_vi": "điều đình, hòa giải, trung gian", "meo": "Vị VƯƠNG (王) mồm RỘNG (口) đi làm (干) điều đình khắp NƠI (方).", "reading": "あっ", "vocab": [("斡旋", "あっせん", "môi giới, trung gian, điều giải"), ("斡旋金", "あっせんきん", "tiền hoa hồng môi giới")]},
    "韓": {"viet": "HÀN", "meaning_vi": "Tên nước ngày xưa. Là một nước nhà Chu [周] phong cho người cùng họ, sau bị nước Tấn [晉] lấy mất, nay thuộc vào vùng tỉnh Thiểm Tây [陝西].", "meo": "", "reading": "から いげた", "vocab": [("韓国", "かんこく", "đại hàn"), ("日韓", "にっかん", "Nhật Hàn")]},
    "廟": {"viet": "MIẾU", "meaning_vi": "Cái miếu (để thờ cúng quỷ thần). Như văn miếu [文廟] đền thờ đức Khổng Tử [孔子].", "meo": "", "reading": "たまや みたまや やしろ", "vocab": [("廟", "びょう", "đền miếu ."), ("古廟", "こびょう", "ngôi miếu cổ .")]},
    "潮": {"viet": "TRIỀU", "meaning_vi": "Nước thủy triều.", "meo": "", "reading": "しお うしお", "vocab": [("潮", "しお", "thủy triều; dòng nước"), ("上潮", "あげしお", "thủy triều lên .")]},
    "嘲": {"viet": "TRÀO", "meaning_vi": "Riễu cợt. Như trào lộng [嘲弄] đùa cợt, trào tiếu [嘲笑] cười nhạo, trào phúng [嘲諷] cười cợt chế nhạo.", "meo": "", "reading": "あざけ.る", "vocab": [("嘲り", "あざけり", "sự nhạo báng"), ("嘲る", "あざける", "chế diễu")]},
    "淡": {"viet": "ĐẠM", "meaning_vi": "Nhạt, sắc hương vị gì nhạt nhẽo đều gọi là đạm. Không ham vinh hoa lợi lộc gọi là đạm bạc [淡泊].", "meo": "", "reading": "あわ.い", "vocab": [("淡々", "たんたん", "vô tư"), ("淡い", "あわい", "nhạt; nhẹ")]},
    "痰": {"viet": "ĐÀM", "meaning_vi": "Đờm.", "meo": "", "reading": "タン", "vocab": [("痰", "たん", "đờm ."), ("去痰", "きょたん", "sự khạc")]},
    "溝": {"viet": "CÂU", "meaning_vi": "Cái ngòi (rãnh); ngòi nước qua các cánh đồng.", "meo": "", "reading": "みぞ", "vocab": [("溝", "みぞ", "khoảng cách"), ("側溝", "そっこう", "máng nước")]},
    "昏": {"viet": "HÔN", "meaning_vi": "Tối. Như hoàng hôn [黃昏] mờ mờ tối, hôn dạ [昏夜] đêm tối, v.v. Lý Thương Ẩn [李商隱] : Tịch dương vô hạn hảo, Chỉ thị cận hoàng hôn [夕陽無限好, 只是近黃昏] (Đăng Lạc Du nguyên [登樂遊原]) Nắng chiều đẹp vô hạn, Chỉ (tiếc) là đã gần hoàng hôn. Quách Tấn dịch thơ : Tịch dương cảnh đẹp vô ngần, Riêng thương chiếc bóng đã gần hoàng hôn.", "meo": "", "reading": "くら.い くれ", "vocab": [("昏倒", "こんとう", "sự ngất đi"), ("昏睡", "こんすい", "sự hôn mê .")]},
    "罠": {"viet": "", "meaning_vi": "trap, snare", "meo": "", "reading": "わな あみ", "vocab": [("罠", "わな", "bẫy; cái bẫy"), ("罠にかかる", "わなにかかる", "mắc bẫy .")]},
    "抵": {"viet": "ĐỂ, CHỈ", "meaning_vi": "Mạo phạm. Như để xúc [抵觸] chọc chạm đến.", "meo": "", "reading": "テイ", "vocab": [("大抵", "たいてい", "đại để; nói chung; thường"), ("抵当", "ていとう", "cầm đồ")]},
    "邸": {"viet": "ĐỂ", "meaning_vi": "Cái nhà cho các nước chư hầu đến chầu ở.", "meo": "", "reading": "やしき", "vocab": [("邸", "やしき", "lâu đài"), ("公邸", "こうてい", "dinh thự của quan chức cấp cao để làm việc công")]},
    "曖": {"viet": "ÁI", "meaning_vi": "Yểm ái [晻曖] mờ mịt.", "meo": "", "reading": "くら.い", "vocab": [("曖昧", "あいまい", "mơ hồ; khó hiểu; lờ mờ; mập mờ"), ("曖昧さ", "あいまいさ", "Sự nhập nhằng; sự lờ mờ; khó hiểu .")]},
    "賄": {"viet": "HỐI", "meaning_vi": "Của. Như hóa hối [貨賄] của cải, vàng ngọc gọi là hóa, vải lụa gọi là hối.", "meo": "", "reading": "まかな.う", "vocab": [("賄い", "まかない", "sự lót ván"), ("賄う", "まかなう", "chịu chi trả")]},
    "髄": {"viet": "TỦY", "meaning_vi": "Xương tủy", "meo": "", "reading": "ズイ", "vocab": [("延髄", "えんずい", "não sau"), ("心髄", "しんずい", "điều huyền bí; điều bí ẩn")]},
    "随": {"viet": "TÙY", "meaning_vi": "Tục dùng như chữ tùy [隨].", "meo": "", "reading": "まにま.に したが.う", "vocab": [("随一", "ずいいち", "đệ nhất"), ("不随", "ふずい", "Chứng liệt .")]},
    "堕": {"viet": "ĐỌA", "meaning_vi": "rơi, sa ngã", "meo": "Thổ (土) ngã xuống (隊)", "reading": "だ", "vocab": [("堕落", "だらく", "sự trụy lạc, sự sa đọa"), ("堕胎", "だたい", "sự phá thai")]},
    "惰": {"viet": "NỌA", "meaning_vi": "Lười biếng. Như nọa tính [惰性] tính lười, du nọa [遊惰] lười biếng ham chơi, không chịu làm ăn.", "meo": "", "reading": "ダ", "vocab": [("惰力", "だりょく", "tính ì"), ("勤惰", "きんだ", "sự dự")]},
    "楕": {"viet": "", "meaning_vi": "ellipse", "meo": "", "reading": "ダ タ", "vocab": [("楕円", "だえん", "hình bầu dục ."), ("楕円形", "だえんけい", "hình elip")]},
    "垂": {"viet": "THÙY", "meaning_vi": "Rủ xuống. Nguyễn Du [阮攸] : Thành nam thùy liễu bất câm phong [城南垂柳不禁風] (Thương Ngô Trúc Chi ca [蒼梧竹枝歌]) Phía nam thành, liễu rủ không đương nổi với gió.", "meo": "", "reading": "た.れる た.らす た.れ -た.れ なんなんと.す", "vocab": [("垂れ", "たれ", "sự treo"), ("下垂", "かすい", "cúi xuống; rũ xuống")]},
    "唾": {"viet": "THÓA", "meaning_vi": "Nhổ, nhổ nước dãi đi gọi là thóa.", "meo": "", "reading": "つば つばき", "vocab": [("唾", "つば", "nước bọt; nước dãi; đờm"), ("唾棄", "だき", "khinh thường")]},
    "剰": {"viet": "THẶNG", "meaning_vi": "Thừa. Như sở thặng vô kỉ [所剰無幾] thửa thừa không mấy nhiều.", "meo": "", "reading": "あまつさえ あま.り あま.る", "vocab": [("剰", "じょう", "sự quá; thặng dư"), ("剰え", "あまっさえ", "ngoài ra")]},
    "嘩": {"viet": "HOA", "meaning_vi": "Cũng như chữ hoa [譁].", "meo": "", "reading": "かまびす.しい", "vocab": [("喧嘩", "けんか", "sự cà khịa; sự cãi cọ; sự tranh chấp; cà khịa; cãi cọ; tranh chấp"), ("口喧嘩", "くちけんか", "cãi nhau; khẩu chiến; đấu khẩu")]},
    "粋": {"viet": "TÚY", "meaning_vi": "Tinh túy", "meo": "", "reading": "いき", "vocab": [("粋", "いき", "tao nhã; sành điệu; mốt; hợp thời trang; lịch thiệp; lịch sự; thanh nhã; sang trọng; bảnh bao"), ("粋な", "すいな", "bảnh .")]},
    "砕": {"viet": "TOÁI", "meaning_vi": "Phá vỡ", "meo": "", "reading": "くだ.く くだ.ける", "vocab": [("砕く", "くだく", "đánh tan"), ("圧砕", "あっさい", "làm tan nát")]},
    "惨": {"viet": "THẢM", "meaning_vi": "Giản thể của chữ [慘].", "meo": "", "reading": "みじ.め いた.む むご.い", "vocab": [("惨い", "むごい", "độc ác"), ("惨め", "みじめ", "đáng thương; đáng buồn")]},
    "疹": {"viet": "CHẨN", "meaning_vi": "ban, mụn (trên da)", "meo": "Bệnh (疒) mà người ta hay tin (㐱) là do chẩn đoán sai.", "reading": "shin", "vocab": [("発疹", "hasshin", "phát ban"), ("薬疹", "yakushin", "ban do thuốc")]},
    "悠": {"viet": "DU", "meaning_vi": "Lo lắng.", "meo": "", "reading": "ユウ", "vocab": [("悠々", "ゆうゆう", "nhàn tản; ung dung"), ("悠久", "ゆうきゅう", "mãi mãi; vĩnh viễn; vĩnh cửu")]},
    "屯": {"viet": "TRUÂN, ĐỒN", "meaning_vi": "Khó. Khó tiến lên được gọi là truân chiên [屯邅]. Còn viết là [迍邅]. Nguyễn Trãi [阮廌] : Bán sinh thế lộ thán truân chiên [半生世路嘆屯邅] (Kí hữu [寄友]) Nửa đời người, than cho đường đời gian nan vất vả.", "meo": "", "reading": "たむろ", "vocab": [("屯", "とん", "một tấn ."), ("屯営", "とんえい", "doanh trại bộ đội .")]},
    "頓": {"viet": "ĐỐN", "meaning_vi": "Cúi xuống sát đất. Như đốn thủ [頓首] lạy rập đầu sát đất.", "meo": "", "reading": "にわか.に とん.と つまず.く とみ.に ぬかずく", "vocab": [("頓に", "とみに", "nhanh"), ("停頓", "ていとん", "sự đình hẳn lại; sự đình trệ hoàn toàn;  sự bế tắc")]},
    "沌": {"viet": "ĐỘN", "meaning_vi": "Hỗn độn [混沌] mờ mịt, nói lúc chưa phân rõ trời đất, nói bóng cái ý chưa khai thông. Còn viết [渾沌].", "meo": "", "reading": "くら.い", "vocab": [("混沌", "こんとん", "Sự lẫn lộn; sự hỗn loạn; sự hỗn độn ."), ("渾沌", "こんとん", "sự lộn xộn; sự hỗn loạn; sự lẫn lộn; lộn xộn; hỗn loạn; lẫn lộn .")]},
    "巳": {"viet": "TỊ", "meaning_vi": "Chi Tị, chi thứ sáu trong mười hai chi.", "meo": "", "reading": "み", "vocab": [("巳", "み", "Tỵ (rắn)"), ("初巳", "はつみ", "Ngày Tỵ đầu tiên trong năm .")]},
    "肥": {"viet": "PHÌ", "meaning_vi": "Béo, phàm các giống động vật thực vật mà có nhiều chất béo gọi là phì. Như phì mĩ [肥美] béo ngậy, ngậy ngon.", "meo": "", "reading": "こ.える こえ こ.やす こ.やし ふと.る", "vocab": [("肥", "こえ", "phân; cứt; phân bón"), ("肥す", "こやす", "làm cho tốt")]},
    "把": {"viet": "BẢ", "meaning_vi": "Cầm. Như bả tí [把臂] cầm tay. Sự gì cầm chắc được gọi là bả ác [把握].", "meo": "", "reading": "ハ ワ", "vocab": [("把", "わ", "bó"), ("一把", "いちわ", "một bó .")]},
    "芭": {"viet": "BA", "meaning_vi": "Cỏ ba, một thứ cỏ thơm.", "meo": "", "reading": "バ ハ", "vocab": []},
    "徐": {"viet": "TỪ", "meaning_vi": "Đi thong thả.", "meo": "", "reading": "おもむ.ろに", "vocab": [("徐々", "じょ々", "dần dần"), ("徐徐", "そろそろ", "dần dần .")]},
    "叙": {"viet": "TỰ", "meaning_vi": "Cũng như chữ [敘].", "meo": "", "reading": "つい.ず ついで", "vocab": [("叙", "じょ", "sự kể lại; sự tường thuật; sự diễn tả; sự mô tả"), ("叙事", "じょじ", "sự kể chuyện")]},
    "塗": {"viet": "ĐỒ, TRÀ", "meaning_vi": "Bùn bửn. Đãi người tàn ác gọi là đồ thán [塗炭] lầm than.", "meo": "", "reading": "ぬ.る ぬ.り まみ.れる", "vocab": [("塗る", "ぬる", "chét"), ("上塗", "うわぬり", "sự kết thúc")]},
    "李": {"viet": "LÍ", "meaning_vi": "Cây mận.", "meo": "", "reading": "すもも", "vocab": [("李", "り", "sửa"), ("行李", "こうり", "va li")]},
    "萎": {"viet": "NUY", "meaning_vi": "Héo, cây cỏ héo.", "meo": "", "reading": "な しお.れる しな.びる しぼ.む な.える", "vocab": [("萎む", "しぼむ", "chắc chắn; ổn định"), ("萎える", "なえる", "làm héo")]},
    "痢": {"viet": "LỊ", "meaning_vi": "Bệnh lị, quặn đau bụng lại đi ra ngoài, rặn mãi mới ra ít phân máu hay ít mũi là lị.", "meo": "", "reading": "リ", "vocab": [("痢", "り", "bệnh ỉa chảy; bệnh tiêu chảy [y học]"), ("下痢", "げり", "bệnh đi ỉa; bệnh tiêu chảy; ỉa chảy")]},
    "愁": {"viet": "SẦU", "meaning_vi": "Sầu, lo, buồn thảm.", "meo": "", "reading": "うれ.える うれ.い", "vocab": [("愁い", "うれい", "nỗi u sầu; buồn bã; ủ dột; buồn rầu; buồn sầu; rầu rĩ"), ("愁傷", "しゅうしょう", "nỗi đau buồn")]},
    "萩": {"viet": "THU", "meaning_vi": "(Thực vật) Một loại cỏ ngải, mọc ở bờ nước, đất cát, cao khoảng 3 thước, mùa hè nở hoa xanh.", "meo": "", "reading": "はぎ", "vocab": [("萩", "はぎ", "chân"), ("萩属", "はぎぞく", "cây hồ chì")]},
    "嘆": {"viet": "THÁN", "meaning_vi": "Than, thở dài. Như thán tức [嘆息] than thở.", "meo": "", "reading": "なげ.く なげ.かわしい", "vocab": [("嘆き", "なげき", "nỗi đau; nỗi buồn"), ("嘆く", "なげく", "thở dài; kêu than; than thở")]},
    "謹": {"viet": "CẨN", "meaning_vi": "Cẩn thận, cẩn trọng, nghĩa là làm việc để ý kỹ lưỡng không dám coi thường.", "meo": "", "reading": "つつし.む", "vocab": [("謹む", "つつしむ", "hân hạnh; khiêm tốn; kính cẩn"), ("謹厳", "きんげん", "nghiêm nghị")]},
    "僅": {"viet": "CẬN", "meaning_vi": "chỉ, ít, bé", "meo": "Chữ NHÂN đứng cạnh KỶ (mấy) cho thấy chỉ một KỶ người thôi, tức là rất ÍT.", "reading": "わず", "vocab": [("僅か", "わずか", "ít, chút ít, một chút"), ("僅少", "きんしょう", "số lượng nhỏ, ít ỏi")]},
    "矛": {"viet": "MÂU", "meaning_vi": "Cái giáo, một thứ đồ binh cán dài có mũi nhọn.", "meo": "", "reading": "ほこ", "vocab": [("矛", "ほこ", "mâu"), ("矛先", "ほこさき", "mũi mâu; mũi dao")]},
    "霧": {"viet": "VỤ", "meaning_vi": "Sương mù. Nguyên nhân cũng như mây, xa đất là vân [雲] mây, gần đất là vụ [霧] mù. Đỗ Phủ [杜甫] : Hương vụ vân hoàn thấp [香霧雲鬟濕] (Nguyệt dạ [月夜]) Sương thơm làm ướt mái tóc mai. Tản Đà dịch thơ : Sương sa thơm ướt mái đầu.", "meo": "", "reading": "きり", "vocab": [("霧", "きり", "cỏ mọc lại"), ("霧吹", "きりふき", "bình phun")]},
    "是": {"viet": "THỊ", "meaning_vi": "Phải, điều gì ai cũng công nhận là phải gọi là thị. Cái phương châm của chánh trị gọi là quốc thị [國是].", "meo": "", "reading": "これ この ここ", "vocab": [("如是", "にょぜ", "như thế"), ("店是", "てんぜ", "Chính sách cửa hàng .")]},
    "堤": {"viet": "ĐÊ", "meaning_vi": "Cái đê.", "meo": "", "reading": "つつみ", "vocab": [("堤", "つつみ", "bờ đê"), ("堰堤", "えんてい", "đê")]},
    "提": {"viet": "ĐỀ, THÌ, ĐỂ", "meaning_vi": "Nâng lên, nâng đỡ, phàm dắt cho lên trên, kéo cho tiến lên đều gọi là đề. Như đề huề [提攜] dắt díu, đề bạt [提拔] cất nhắc, v.v.", "meo": "", "reading": "さ.げる", "vocab": [("上提", "うえひさげ", "sự bày ra"), ("提供", "ていきょう", "chào giá")]},
    "匙": {"viet": "THI", "meaning_vi": "Cái thìa.", "meo": "", "reading": "さじ", "vocab": [("匙", "さじ", "cái muỗng"), ("小匙", "こさじ", "thìa cà phê .")]},
    "稿": {"viet": "CẢO", "meaning_vi": "Rơm rạ. Lấy rơm rạ làm đệm gọi là cảo tiến [稿薦].", "meo": "", "reading": "わら したがき", "vocab": [("稿", "こう", "bản thảo; bản nháp ."), ("稿人", "こうじん", "hình nộm bằng rơm; bù nhìn rơm .")]},
    "縞": {"viet": "CẢO", "meaning_vi": "The mộc mỏng, đơn sơ.", "meo": "", "reading": "しま しろぎぬ", "vocab": [("縞", "しま", "kẻ hoa ."), ("縞馬", "しまうま", "ngựa vằn")]},
    "嵩": {"viet": "TUNG", "meaning_vi": "Núi Tung. Hán Võ đế [漢武帝] lên chơi núi Tung Sơn [嵩山], quan, lính đều nghe tiếng xưng hô vạn tuế đến ba lần. Vì thế ngày nay đi chúc thọ gọi là tung chúc [嵩祝].", "meo": "", "reading": "かさ かさ.む たか.い", "vocab": [("嵩", "かさ", "trọng tải hàng hoá; hàng hoá"), ("嵩む", "かさむ", "sự tăng")]},
    "藁": {"viet": "CẢO", "meaning_vi": "Cây khô.", "meo": "", "reading": "わら", "vocab": [("藁", "わら", "rơm"), ("寝藁", "ねわら", "Ổ rơm (thường dùng để súc vật ngủ)")]},
    "膏": {"viet": "CAO, CÁO", "meaning_vi": "Mỡ. Mỡ miếng gọi là chi [脂], mỡ nước gọi là cao [膏]. Như lan cao [蘭膏] dầu thơm, cao mộc [膏沐] sáp bôi, v.v.", "meo": "", "reading": "あぶら", "vocab": [("石膏", "せっこう", "trát vữa ; trát thạch cao"), ("膏薬", "こうやく", "thuốc cao")]},
    "矯": {"viet": "KIỂU", "meaning_vi": "Nắn thẳng, cái gì lầm lỗi sửa lại cho phải gọi là kiểu chính [矯正].", "meo": "", "reading": "た.める", "vocab": [("奇矯", "ききょう", "kỳ cục; kỳ quặc; lập dị; quái gở"), ("矯める", "ためる", "làm thẳng ra; sửa lại; sửa chữa; uốn nắn; cải tiến chất lượng")]},
    "蕎": {"viet": "KIỀU", "meaning_vi": "Kiều mạch [蕎麥] lúa tám đen.", "meo": "", "reading": "そば", "vocab": [("蕎麦", "そば", "mỳ soba; mỳ từ kiều mạch"), ("蕎麦屋", "そばや", "nhà hàng chuyên mỳ soba .")]},
    "仙": {"viet": "TIÊN", "meaning_vi": "Tiên. Nhà đạo sĩ luyện thuốc trừ cơm tu hành, cầu cho sống mãi không chết gọi là tiên [仙].", "meo": "", "reading": "セン セント", "vocab": [("仙", "せん", "tiên nhân"), ("仙人", "せんにん", "tiên nhân .")]},
    "辿": {"viet": "", "meaning_vi": "follow (road), pursue", "meo": "", "reading": "たど.る たどり", "vocab": [("辿る", "たどる", "theo dấu; lần theo")]},
    "巾": {"viet": "CÂN", "meaning_vi": "Cái khăn.", "meo": "", "reading": "おお.い ちきり きれ", "vocab": [("巾", "はば", "khăn ăn"), ("値巾", "ねはば", "khoảng dao động của giá cả .")]},
    "吊": {"viet": "ĐIẾU", "meaning_vi": "Cũng như chữ điếu [弔]. Nguyễn Du [阮攸] : Giang biên hà xứ điếu trinh hồn [江邊何處吊貞魂] (Tam liệt miếu [三烈廟]) Bên sông, đâu nơi viếng hồn trinh ?", "meo": "", "reading": "つ.る つる.す", "vocab": [("吊す", "つるす", "sự cúi xuống"), ("吊り", "つり", "sự đánh cá")]},
    "稀": {"viet": "HI", "meaning_vi": "Thưa thớt. Như địa quảng nhân hi [地廣人稀] đất rộng người thưa.", "meo": "", "reading": "まれ まばら", "vocab": [("稀", "まれ", "hiếm có; ít có"), ("稀世", "きせい", "hiếm")]},
    "逓": {"viet": "ĐỆ", "meaning_vi": "Đệ trình, gửi đi", "meo": "", "reading": "かわ.る たがいに", "vocab": [("逓伝", "ていでん", "rơ le ."), ("逓信", "ていしん", "thông tin")]},
    "凧": {"viet": "(DIỀU)", "meaning_vi": "Con diều.", "meo": "", "reading": "いかのぼり たこ", "vocab": [("凧", "たこ", "cái diều ."), ("凧揚げ", "たこあげ", "thả diều")]},
    "凪": {"viet": "", "meaning_vi": "lull, calm, (kokuji)", "meo": "", "reading": "なぎ な.ぐ", "vocab": [("凪", "なぎ", "Sự tĩnh lặng; sự yên lặng; trời yên biển lặng"), ("凪ぐ", "なぐ", "yếu dần")]},
    "颯": {"viet": "TÁP", "meaning_vi": "Tiếng gió thổi vèo vèo. Lý Thương Ẩn [李商隱] : Táp táp đông phong tế vũ lai [颯颯東風細雨來] (Vô đề [無題]) Rào rạt gió đông (gió xuân); mưa nhỏ (mưa phùn) đến.", "meo": "", "reading": "さっ.と", "vocab": [("颯と", "さっと", "một cách êm ả; một cách trôi chảy"), ("颯爽", "さっそう", "dũng cảm; hào hiệp")]},
    "此": {"viet": "THỬ", "meaning_vi": "Ấy, bên ấy, đối lại với chữ bỉ [彼].", "meo": "", "reading": "これ この ここ", "vocab": [("此の", "この", "này"), ("此れ", "これ", "cái này; đây")]},
    "雌": {"viet": "THƯ", "meaning_vi": "Con mái, loài có lông cánh thuộc về tính âm (giống cái) gọi là thư, con thú cái cũng gọi là thư.", "meo": "", "reading": "め- めす めん", "vocab": [("雌", "めす", "con cái; cái"), ("雌伏", "しふく", "phần bị che khuất")]},
    "砦": {"viet": "TRẠI", "meaning_vi": "Ở núi lấy gỗ ken chung quanh làm hàng rào gọi là trại.", "meo": "", "reading": "とりで", "vocab": [("砦", "とりで", "Pháo đài ."), ("城砦", "しろとりで", "pháo đài")]},
    "柴": {"viet": "SÀI, TÍ", "meaning_vi": "Củi.", "meo": "", "reading": "しば", "vocab": [("柴", "しば", "bụi cây")]},
    "淀": {"viet": "ĐIẾN", "meaning_vi": "Chỗ nước nông. Như hồ ao, v.v.", "meo": "", "reading": "よど.む", "vocab": [("淀む", "よどむ", "đọng")]},
    "錠": {"viet": "ĐĨNH", "meaning_vi": "Cái choé, một thứ đồ làm bằng loài kim, có chân, để dâng các đồ nấu chín.", "meo": "", "reading": "ジョウ", "vocab": [("錠", "じょう", "món tóc"), ("一錠", "いちじょう", "một khay")]},
    "綻": {"viet": "TRÁN", "meaning_vi": "Đường khâu áo. Như thoát trán [脫綻] áo sứt chỉ.", "meo": "", "reading": "ほころ.びる", "vocab": [("綻び", "ほころび", "nước mắt"), ("綻びる", "ほころびる", "rách; bục; hỏng")]},
    "婿": {"viet": "TẾ", "meaning_vi": "Như chữ tế [壻]. Vương Xương Linh [王昌齡] : Hốt kiến mạch đầu dương liễu sắc, Hối giao phu tế mịch phong hầu [忽見陌頭楊柳色, 悔教夫婿覓封侯] (Khuê oán [閨怨]) Chợt thấy sắc cây dương liễu ở đầu đường, Hối tiếc đã khuyên chồng ra đi cầu mong được phong tước hầu.", "meo": "", "reading": "むこ", "vocab": [("婿", "むこ", "con rể ."), ("令婿", "れいせい", "êm đềm")]},
    "礎": {"viet": "SỞ", "meaning_vi": "Đá tảng, dùng kê chân cột.", "meo": "", "reading": "いしずえ", "vocab": [("礎", "いしずえ", "đá lót nền; nền; nền tảng"), ("基礎", "きそ", "căn bản")]},
    "兜": {"viet": "ĐÂU", "meaning_vi": "Đâu mâu [兜鍪] cái mũ trụ. Cái mũ lúc ra đánh trận thì đội.", "meo": "", "reading": "かぶと", "vocab": [("兜", "かぶと", "mũ sắt"), ("兜虫", "かぶとちゅう", "cái chày")]},
    "仰": {"viet": "NGƯỠNG, NHẠNG", "meaning_vi": "Ngửa, ngửa mặt lên gọi là ngưỡng.", "meo": "", "reading": "あお.ぐ おお.せ お.っしゃる おっしゃ.る", "vocab": [("仰ぐ", "あおぐ", "lệ thuộc; phụ thuộc"), ("仰せ", "おおせ", "lệnh; mệnh lệnh")]},
    "抑": {"viet": "ỨC", "meaning_vi": "Đè nén. Như ức chế [抑制].", "meo": "", "reading": "おさ.える", "vocab": [("抑", "そもそも", "đầu tiên; ngay từ ban đầu"), ("抑え", "おさえ", "quyền hành")]},
    "叩": {"viet": "KHẤU", "meaning_vi": "Gõ. Như khấu môn [叩門] gõ cửa, khấu quan [叩關] gõ cửa quan, v.v.", "meo": "", "reading": "たた.く はた.く すぎ", "vocab": [("叩き", "はたき", "cái phất trần; chổi lông"), ("叩く", "たたく", "bịch")]},
    "卯": {"viet": "MÃO, MẸO", "meaning_vi": "Chi Mão. Chi thứ tư trong 12 chi. Từ năm giờ sáng đến bảy giờ sáng là giờ Mão.", "meo": "", "reading": "う", "vocab": []},
    "柳": {"viet": "LIỄU", "meaning_vi": "Cây liễu. Nguyễn Du [阮攸] : Thành nam thùy liễu bất câm phong [城南垂柳不禁風] (Thương Ngô Trúc Chi ca [蒼梧竹枝歌]) Thành nam liễu rủ khôn ngăn gió.", "meo": "", "reading": "やなぎ", "vocab": [("柳", "やなぎ", "liễu; cây liễu ."), ("川柳", "せんりゅう", "bài thơ hài hước viết ở thể loại haiku .")]},
    "曽": {"viet": "TẰNG", "meaning_vi": "Một dạng của chữ tằng [曾].", "meo": "", "reading": "かつ かつて すなわち", "vocab": [("曽孫", "そうそん", "chắt"), ("曽孫", "ひいまご", "Chắt")]},
    "僧": {"viet": "TĂNG", "meaning_vi": "Sư nam, đàn ông đi tu đạo Phật 佛 gọi là tăng. Nguyên tiếng Phạn gọi là Tăng già [僧伽] (sangha) nghĩa là một đoàn thể đệ tử Phật. Trong luật định bốn vị sư trở lên mới gọi là Tăng già.", "meo": "", "reading": "ソウ", "vocab": [("僧", "そう", "nhà sư"), ("仏僧", "ぶっそう", "nhà sư; tăng lữ .")]},
    "噌": {"viet": "TẰNG", "meaning_vi": "Ồn ào; ầm ĩ, náo nhiệt.", "meo": "", "reading": "かまびす.しい", "vocab": [("味噌", "みそ", "điểm chính; điểm chủ chốt"), ("弱味噌", "じゃくみそ", "người yếu ớt")]},
    "促": {"viet": "XÚC", "meaning_vi": "Gặt, sự cần kíp đến nơi gọi là xúc. Như cấp xúc [急促] vội gấp, đoản xúc [短促] ngắn gặt, suyễn xúc [喘促] thở gặt, v.v.", "meo": "", "reading": "うなが.す", "vocab": [("促す", "うながす", "thúc giục; thúc đẩy; xúc tiến; kích thích; động viên; khuyến khích; giục giã; giục; kêu gọi"), ("催促", "さいそく", "sự thúc giục; sự giục giã .")]},
    "捉": {"viet": "TRÓC", "meaning_vi": "bắt, nắm bắt, tóm lấy", "meo": "Bàn tay (扌) tóm lấy con chó (犬) rất nhanh, tróc lấy nó!", "reading": "とら", "vocab": [("捉える", "とらえる", "bắt giữ, tóm lấy, nắm bắt"), ("捉まる", "つかまる", "bị bắt, bị tóm, bám vào")]},
    "辰": {"viet": "THẦN, THÌN", "meaning_vi": "Chi Thần (ta đọc là Thìn); chi thứ năm trong 12 chi. Từ bảy giờ sáng cho đến chín giờ sáng gọi là giờ Thìn.", "meo": "", "reading": "たつ", "vocab": [("星辰", "せいしん", "thiên thể"), ("誕辰", "たんしん", "ngày sinh; lễ sinh nhật")]},
    "賑": {"viet": "CHẨN", "meaning_vi": "Giàu.", "meo": "", "reading": "にぎ.わい にぎ.やか にぎ.わす にぎ.わう", "vocab": [("賑やか", "にぎやか", "sôi nổi; náo nhiệt; sống động; huyên náo"), ("賑わい", "にぎわい", "Sự thịnh vượng; sự nhộn nhịp")]},
    "唇": {"viet": "THẦN", "meaning_vi": "Tục dùng như chữ thần [脣].", "meo": "", "reading": "くちびる", "vocab": [("唇", "くちびる", "môi"), ("上唇", "うわくちびる", "môi trên")]},
    "辱": {"viet": "NHỤC", "meaning_vi": "làm nhục, sỉ nhục", "meo": "Vô (vô nghĩa) vào người trên (尸) lại còn lạy (寸) thì đúng là nhục", "reading": "はずかしめる", "vocab": [("侮辱", "ぶじょく", "sỉ nhục, lăng mạ"), ("雪辱", "せつじょく", "rửa hận, phục thù")]},
    "孔": {"viet": "KHỔNG", "meaning_vi": "Rất, lắm. Như mưu phủ khổng đa [謀夫孔多] người mưu rất nhiều.", "meo": "", "reading": "あな", "vocab": [("孔", "あな", "lỗ"), ("多孔", "たこう", "nhiều hang động")]},
    "踪": {"viet": "TUNG", "meaning_vi": "Cùng nghĩa với chữ [蹤].", "meo": "", "reading": "あと", "vocab": [("失踪", "しっそう", "sự biến đi")]},
    "綜": {"viet": "TỐNG, TÔNG", "meaning_vi": "Đem dệt sợi nọ với sợi kia gọi là tống. Vì thế sự gì lẫn lộn với nhau gọi là thác tống [錯綜].", "meo": "", "reading": "おさ.める す.べる", "vocab": [("綜合", "そうごう", "sự tổng hợp"), ("綜覧", "そうらん", "sự trông nom")]},
    "崇": {"viet": "SÙNG", "meaning_vi": "Cao. Như sùng san [崇山] núi cao.", "meo": "", "reading": "あが.める", "vocab": [("尊崇", "そんすう", "sự tôn kính; lòng sùng kính"), ("崇める", "あがめる", "tôn kính")]},
    "捺": {"viet": "NẠI", "meaning_vi": "Đè ép, lấy tay ấn mạnh gọi là nại.", "meo": "", "reading": "さ.す お.す", "vocab": [("捺す", "おす", "đóng ."), ("捺印", "なついん", "con dấu")]},
    "斎": {"viet": "TRAI", "meaning_vi": "Trai giới", "meo": "", "reading": "とき つつし.む ものいみ い.む いわ.う いつ.く", "vocab": [("斎", "とき", "sự kiêng"), ("斎く", "いつく", "sự thờ cúng")]},
    "尉": {"viet": "ÚY, UẤT", "meaning_vi": "Quan úy, các quan coi ngục và bắt trộm giặc đời xưa đều gọi là úy. Như đình úy [廷尉], huyện úy [縣尉] đều lấy cái nghĩa trừ kẻ gian cho dân yên cả.", "meo": "", "reading": "イ ジョウ", "vocab": [("尉", "じょう", "hàng"), ("三尉", "さんじょう", "sự tán thành")]},
    "慰": {"viet": "ÚY, ỦY", "meaning_vi": "Yên. Như hân úy [欣慰] yên vui, úy lạo [慰勞] yên ủi. $ Ta quen đọc là ủy.", "meo": "", "reading": "なぐさ.める なぐさ.む", "vocab": [("慰み", "なぐさみ", "sự an ủi"), ("慰む", "なぐさむ", "an ủi; động viên; giải trí; vui chơi")]},
    "俵": {"viet": "BIỂU", "meaning_vi": "Chia cho. Ta đem cái gì cho ai gọi là biếu, có nhẽ cũng noi chữ này.", "meo": "", "reading": "たわら", "vocab": [("俵", "たわら", "bì cỏ; túi rơm; bao bì làm bằng rơm"), ("一俵", "いっぴょう", "đầy bao; bao")]},
    "征": {"viet": "CHINH", "meaning_vi": "Đi xa. Như chinh phu [征夫] người đi đánh giặc phương xa, chinh hồng [征鴻] con chim hồng bay xa, chỉ người đi xa.", "meo": "", "reading": "セイ", "vocab": [("征伐", "せいばつ", "sự chinh phạt; sự thám hiểm ."), ("出征", "しゅっせい", "sự ra trận; việc ra trận .")]},
    "涎": {"viet": "TIÊN, DIỆN, DUYÊN", "meaning_vi": "Nước dãi. Người ta thấy thức ăn ngon thì thèm chảy dãi ra, cho nên ý thích cái gì cũng gọi là thùy tiên [垂涎] thèm nhỏ dãi.", "meo": "", "reading": "よだれ", "vocab": [("涎", "よだれ", "nước dãi"), ("垂涎", "すいぜん", "sự thèm muốn")]},
    "吾": {"viet": "NGÔ", "meaning_vi": "Ta, tới mình mà nói gọi là ngô. Nhân người mà nói gọi là ngã [我]. Như ngã thiện dưỡng ngô hạo nhiên chi khí [我善養吾浩然之氣] ta khéo nuôi cái khí hạo nhiên của ta.", "meo": "", "reading": "われ わが- あ-", "vocab": [("吾", "われ", "tôi"), ("吾が", "わが", "của tôi")]},
    "悟": {"viet": "NGỘ", "meaning_vi": "Tỏ ngộ, biết. Trong lòng hiểu thấu gọi là ngộ, đọc sách hiểu được ý hay gọi là ngộ tính [悟性].", "meo": "", "reading": "さと.る", "vocab": [("悟り", "さとり", "sự khai sáng; sự giác ngộ ."), ("悟る", "さとる", "lính hội; giác ngộ; hiểu được; nhận thức được .")]},
    "陪": {"viet": "BỒI", "meaning_vi": "hầu, kèm theo, phụ", "meo": "Đất (土) gấp đôi (倍) thì cần người bồi đắp.", "reading": "ばい", "vocab": [("陪席", "ばいせき", "dự, ngồi cùng"), ("陪審", "ばいしん", "bồi thẩm đoàn")]},
    "培": {"viet": "BỒI, BẬU", "meaning_vi": "Vun bón.", "meo": "", "reading": "つちか.う", "vocab": [("培う", "つちかう", "bồi dưỡng; vun xới"), ("栽培", "さいばい", "sự cày cấy")]},
    "賠": {"viet": "BỒI", "meaning_vi": "Đền trả. Như bồi thường tổn thất [賠償損失] đền bù chỗ thiệt hại.", "meo": "", "reading": "バイ", "vocab": [("賠償", "ばいしょう", "sự bồi thường ."), ("賠償する", "ばいしょうする", "báo đền")]},
    "剖": {"viet": "PHẨU, PHẪU", "meaning_vi": "Phanh ra. Như phẩu giải [剖解] mổ sả.", "meo": "", "reading": "ボウ", "vocab": [("剖", "ぼう", "sự phân tách; sự chia ra ."), ("剖検", "ぼうけん", "sự mổ xẻ phân tích")]},
    "菩": {"viet": "BỒ", "meaning_vi": "Bồ đề [菩提] dịch âm chữ Phạm bodhi, nghĩa là tỏ biết lẽ chân chính. Tàu dịch là chính giác [正覺].", "meo": "", "reading": "ボ", "vocab": [("菩提", "ぼだい", "bồ đề ."), ("菩薩", "ぼさつ", "bồ tát .")]},
    "刈": {"viet": "NGẢI", "meaning_vi": "Cắt cỏ.", "meo": "", "reading": "か.る", "vocab": [("刈", "かり", "sự cắt"), ("刈る", "かる", "gặt; cắt; tỉa")]},
    "壮": {"viet": "TRÁNG", "meaning_vi": "Giản thể của chữ 壯", "meo": "", "reading": "さかん", "vocab": [("壮丁", "そうてい", "tuổi trẻ"), ("壮健", "そうけん", "khoẻ mạnh; tráng kiện")]},
    "寸": {"viet": "THỐN", "meaning_vi": "thốn (đơn vị đo)", "meo": "Một chấm (、) trên đầu người (才) bị trượt xuống (下) còn một thốn.", "reading": "すん", "vocab": [("寸法", "すんぽう", "kích thước, số đo"), ("寸前", "すんぜん", "gần, suýt soát")]},
    "酎": {"viet": "TRỮU", "meaning_vi": "rượu shochu", "meo": "Rượu (酉) để pha chế (寸) thành đồ uống.", "reading": "ちゅう", "vocab": [("焼酎", "しょうちゅう", "shochu (một loại rượu của Nhật)"), ("酎ハイ", "チューハイ", "chuhai (rượu shochu pha với soda và hương vị)")]},
    "肘": {"viet": "TRỬU", "meaning_vi": "Khuỷu tay. Làm việc mà có người ngăn trở không được thẳng tay làm gọi là xế trửu [掣肘] bó cánh.", "meo": "", "reading": "ひじ", "vocab": [("肘", "ひじ", "cùi chỏ"), ("肘掛", "ひじかけ", "nơi cất vũ khí")]},
    "附": {"viet": "PHỤ", "meaning_vi": "Gắn vào, kèm theo", "meo": "Núi (阝) phải trả tiền (付) khi muốn đi kèm, đi 'phụ' với người khác.", "reading": "ふ", "vocab": [("付属品", "ふぞくひん", "Phụ kiện"), ("付く", "つく", "Dính, gắn liền")]},
    "腐": {"viet": "HỦ", "meaning_vi": "Thối nát.", "meo": "", "reading": "くさ.る -くさ.る くさ.れる くさ.れ くさ.らす くさ.す", "vocab": [("腐り", "くさり", "sự mục"), ("腐る", "くさる", "buồn chán")]},
    "狩": {"viet": "THÚ", "meaning_vi": "Lễ đi săn mùa đông.", "meo": "", "reading": "か.る か.り -が.り", "vocab": [("狩", "かり", "cuộc đi săn"), ("狩り", "かり", "gom; hái; lượm; nhặt")]},
    "献": {"viet": "HIẾN", "meaning_vi": "Giản thể của chữ [獻].", "meo": "", "reading": "たてまつ.る", "vocab": [("献上", "けんじょう", "sự dâng tặng; sự dâng hiến; sự cống tiến; sự cung tiến; dâng tặng; dâng hiến; cống tiến; cung tiến"), ("献呈", "けんてい", "sự bày ra")]},
    "彦": {"viet": "NGẠN", "meaning_vi": "Giản thể của chữ [彥].", "meo": "", "reading": "ひこ", "vocab": [("俊彦", "としひこ", "chính xác")]},
    "諺": {"viet": "NGẠN", "meaning_vi": "Lời tục ngữ. Như ngạn ngôn [諺言], ngạn ngữ [諺語].", "meo": "", "reading": "ことわざ", "vocab": [("諺", "ことわざ", "tục ngữ"), ("俗諺", "ぞくげん", "tục ngữ")]},
    "袖": {"viet": "TỤ", "meaning_vi": "Tay áo. Đổng Hiền [董賢] được vua yêu, nằm gối vào tay áo vua Hán Ai đế [漢哀帝] mà ngủ, khi vua dậy trước, mới dứt tay áo mà dậy, vì thế bọn đàn ông được vua yêu gọi là đoạn tụ [斷袖].", "meo": "", "reading": "そで", "vocab": [("袖", "そで", "ống tay áo"), ("半袖", "はんそで", "áo ngắn tay; áo cộc tay")]},
    "笛": {"viet": "ĐỊCH", "meaning_vi": "Cái sáo. Đời sau gọi thứ sáo thổi dọc là cái tiêu [蕭], thứ thổi ngang là địch [笛]. Nguyễn Du [阮攸] : Đoản địch thanh thanh minh nguyệt trung [短笛聲聲明月中] (Thăng Long [昇龍]) Sáo vẳng từng hồi dưới sáng trăng.", "meo": "", "reading": "ふえ", "vocab": [("笛", "ふえ", "cái còi; cái sáo"), ("口笛", "くちぶえ", "còi")]},
    "抽": {"viet": "TRỪU", "meaning_vi": "Kéo ra. Như trừu thủy cơ khí [抽水機器] cái máy kéo nước.", "meo": "", "reading": "ひき-", "vocab": [("抽出", "ちゅうしゅつ", "trích ra; rút ra (từ một chất lỏng.v.v...); rút ra một mẫu từ trong tập hợp (thống kê); sự chiết xuất ."), ("抽せん", "ちゅうせん", "cuộc xổ số")]},
    "柚": {"viet": "DỮU, TRỤC", "meaning_vi": "Cây dữu (cây quýt quả nhỏ). Một thứ cây có quả ăn được.", "meo": "", "reading": "ゆず", "vocab": []},
    "寅": {"viet": "DẦN", "meaning_vi": "Chi Dần, một chi trong mười hai chi. Từ ba giờ sáng đến năm giờ sáng là giờ Dần.", "meo": "", "reading": "とら", "vocab": [("寅", "とら", "dần"), ("寅年", "とらどし", "năm con hổ .")]},
    "絃": {"viet": "HUYỀN", "meaning_vi": "Dây đàn.", "meo": "", "reading": "いと", "vocab": [("三絃", "さんげん", "đàn Nhật ba dây"), ("五絃", "ごげん", "từ nguyên")]},
    "弦": {"viet": "HUYỀN", "meaning_vi": "Dây cung.", "meo": "", "reading": "つる", "vocab": [("弦", "つる", "dây đàn ."), ("三弦", "さんげん", "đàn Nhật ba dây")]},
    "舷": {"viet": "HUYỀN", "meaning_vi": "Mạn thuyền. Tô Thức [蘇軾] : Khấu huyền nhi ca chi [扣舷而歌之] (Tiền Xích Bích phú [前赤壁賦]) Gõ vào mạn thuyền mà hát.", "meo": "", "reading": "ふなばた", "vocab": [("舷側", "げんそく", "phần mạn tàu nổi trên mặt nước"), ("右舷", "うげん", "mạn phải (của tàu")]},
    "眩": {"viet": "HUYỄN", "meaning_vi": "Hoa mắt.", "meo": "", "reading": "げん.す くるめ.く まぶ.しい くら.む まど.う めま.い まばゆ.い くれ.る ま.う", "vocab": [("眩い", "まばゆい", "lanh lợi"), ("眩しい", "まぶしい", "chói mắt; sáng chói; chói lọi; rạng rỡ; sáng rực; chiếu rực rỡ (mặt trời)")]},
    "蓄": {"viet": "SÚC", "meaning_vi": "Dành chứa. Như súc tích [蓄積] cất chứa, còn có nghĩa là chứa đựng nhiều ý tưởng.", "meo": "", "reading": "たくわ.える", "vocab": [("蓄え", "たくわえ", "sự có nhiều"), ("備蓄", "びちく", "sự tích trữ .")]},
    "牽": {"viet": "KHIÊN, KHẢN", "meaning_vi": "Dắt đi.", "meo": "", "reading": "ひ.く", "vocab": [("牽制", "けんせい", "(từ Mỹ"), ("牽引", "けんいん", "xơ (lanh")]},
    "姥": {"viet": "MỖ, MỤ", "meaning_vi": "Tên đất, cũng như chữ [姆]. Cũng đọc là mụ.", "meo": "", "reading": "うば", "vocab": []},
    "嗜": {"viet": "THỊ", "meaning_vi": "Ham thích. Lê Trắc (*) [黎崱] : Thùy lão thị thư [垂老嗜書] (An Nam chí lược tự [安南志畧序]) Về già càng thích sách vở. $ (*) Lê Mạnh Thát phiên âm là Lê Thực, xem \"Toàn tập Trần Nhân Tông\", in năm 2000, Việt Nam, trang 23.", "meo": "", "reading": "たしな.む たしな.み この.む この.み", "vocab": [("嗜好", "しこう", "vị"), ("嗜眠", "しみん", "trạng thái lịm đi; trạng thái mê mệt")]},
    "孝": {"viet": "HIẾU", "meaning_vi": "Thảo, con thờ cha mẹ hết lòng gọi là hiếu.", "meo": "", "reading": "コウ キョウ", "vocab": [("孝", "こう", "hiếu; sự hiếu thảo"), ("不孝", "ふこう", "bi khổ")]},
    "酵": {"viet": "DIẾU", "meaning_vi": "Men. Meo mốc là chất chảy hâm có chất đường, vì tác dụng hóa học sinh ra vi trùng nổi bọt meo lên thành ra chất chua, gọi là phát diếu [發酵] lên men. Như ủ rượu gây giấm biến ra mùi chua đều là vì thế, cho rượu vào bột cho nó bốc bồng lên cũng gọi là phát diếu.", "meo": "", "reading": "コウ", "vocab": [("酵母", "こうぼ", "men; men bia; mốc; enzim ."), ("発酵", "はっこう", "lên men")]},
    "拷": {"viet": "KHẢO", "meaning_vi": "Đánh tra khảo.", "meo": "", "reading": "ゴウ", "vocab": [("拷問", "ごうもん", "sự tra tấn"), ("拷問具", "ごうもんぐ", "dụng cụ tra tấn")]},
    "渦": {"viet": "QUA, OA", "meaning_vi": "Sông Qua.", "meo": "", "reading": "うず", "vocab": [("渦", "うず", "xoáy"), ("渦中", "かちゅう", "xoáy nước; cơn lốc")]},
    "禍": {"viet": "HỌA", "meaning_vi": "Tai vạ. Nguyễn Trãi [阮廌] : Họa phúc hữu môi phi nhất nhật [禍福有媒非一日] (Quan hải [關海]) Họa phúc đều có nguyên nhân, không phải là chuyện một ngày dấy lên.", "meo": "", "reading": "わざわい", "vocab": [("禍因", "わざわいいん", "<CHTRị> Hạ nghị viện"), ("禍害", "かがい", "xấu")]},
    "鍋": {"viet": "OA", "meaning_vi": "Cái bầu dầu (trong xe có cái bầu dầu đựng dầu mỡ cho trục nó chạy trơn).", "meo": "", "reading": "なべ", "vocab": [("鍋", "なべ", "cái chảo; chảo; xoong"), ("土鍋", "どなべ", "nồi đất .")]},
    "叶": {"viet": "DIỆP, DIẾP, HIỆP", "meaning_vi": "Cổ văn là chữ [協]. Vần cổ lầm lạc, người nhà Tống sửa lại các chữ không hợp vần gọi là hiệp vận [叶韻].", "meo": "", "reading": "かな.える かな.う", "vocab": [("叶う", "かなう", "đáp ứng; phù hợp; thỏa mãn"), ("叶える", "かなえる", "khiến... đạt tới mục đích; đáp ứng nguyện vọng; đáp ứng nhu cầu")]},
    "辻": {"viet": "TỬ", "meaning_vi": "Ngã tư, băng ngang đường; góc đường.", "meo": "", "reading": "つじ", "vocab": [("辻", "つじ", "phố"), ("辻君", "つじくん", "gái giang hồ")]},
    "迅": {"viet": "TẤN", "meaning_vi": "Nhanh chóng. Đi lại vùn vụt, người không lường được gọi là tấn. Như tấn lôi bất cập yểm nhĩ [迅雷不及掩耳] sét đánh không kịp bưng tai.", "meo": "", "reading": "ジン", "vocab": [("奮迅", "ふんじん", "sự lao tới phía trước một cách mãnh liệt ."), ("迅速", "じんそく", "mau lẹ; nhanh chóng")]},
    "訊": {"viet": "TẤN", "meaning_vi": "Hỏi. Kẻ trên hỏi kẻ dưới là tấn.", "meo": "", "reading": "き.く と.う たず.ねる", "vocab": [("訊く", "きく", "hỏi"), ("訊問", "じんもん", "sự hỏi dò")]},
    "猿": {"viet": "VIÊN", "meaning_vi": "Con vượn.  Nguyễn Trãi [阮廌] : Viên hạc tiêu điều ý phỉ câm [猿鶴蕭條意匪禁] (Khất nhân họa Côn Sơn đồ [乞人畫崑山圖]) Vượn và hạc tiêu điều, cảm xúc khó cầm.", "meo": "", "reading": "さる", "vocab": [("猿", "さる", "khỉ"), ("山猿", "やまざる", "con khỉ")]},
    "還": {"viet": "HOÀN, TOÀN", "meaning_vi": "Trở lại, về. Đã đi rồi trở lại gọi là hoàn. Như hoàn gia [還家] trở về nhà. Vương An Thạch [王安石] : Minh nguyệt hà thời chiếu ngã hoàn [明月何時照我還] (Bạc thuyền Qua Châu [泊船瓜州]) Bao giờ trăng sáng soi đường ta về ? Đào Trinh Nhất dịch thơ : Đường về nào biết bao giờ trăng soi.", "meo": "", "reading": "かえ.る", "vocab": [("還る", "かえる", "sự trở lại"), ("還付", "かんぷ", "sự trở lại")]},
    "亜": {"viet": "Á", "meaning_vi": "Một dạng của chữ á [亞].", "meo": "", "reading": "つ.ぐ", "vocab": [("亜属", "あぞく", "phân nhóm"), ("東亜", "とうあ", "đông á .")]},
    "唖": {"viet": "Á", "meaning_vi": "Người câm điếc", "meo": "", "reading": "おし", "vocab": [("唖", "あ", "câm"), ("唖", "あく", "câm .")]},
    "壷": {"viet": "HỒ", "meaning_vi": "Cái ấm; bình đựng.", "meo": "", "reading": "つぼ", "vocab": [("壷", "つぼ", "cái bình ."), ("油壷", "あぶらつぼ", "thùng dầu")]},
    "栗": {"viet": "LẬT", "meaning_vi": "Cây lật (cây dẻ); nhân nó ăn được.", "meo": "", "reading": "くり おののく", "vocab": [("栗", "くり", "hạt dẻ"), ("団栗", "どんぐり", "quả đầu")]},
    "慄": {"viet": "LẬT", "meaning_vi": "Sợ run.", "meo": "", "reading": "ふる.える おそ.れる おのの.く", "vocab": [("戦慄", "せんりつ", "sự rùng mình"), ("慄然", "りつぜん", "sự khiếp")]},
    "遷": {"viet": "THIÊN", "meaning_vi": "dời, chuyển", "meo": "Bộ quẹt (辶) chạy theo sau người có Mũ Miện (宀) đội trên đầu, hai người (イイ) đang di chuyển, THIÊN di.", "reading": "せん", "vocab": [("変遷", "へんせん", "biến thiên, thay đổi"), ("遷都", "せんと", "dời đô, chuyển đô")]},
    "漂": {"viet": "PHIÊU, PHIẾU", "meaning_vi": "Nổi. Như phiêu lưu [漂流] trôi nổi, phiêu bạc [漂泊] trôi giạt, v.v.", "meo": "", "reading": "ただよ.う", "vocab": [("漂々", "ひょうひょう", "sự thảnh thơi; sự thoải mái"), ("漂う", "ただよう", "dạt dào; tràn trề; đầy rẫy")]},
    "剽": {"viet": "PHIÊU", "meaning_vi": "ăn cướp, đạo văn", "meo": "Bên trái là chữ Tây, bên phải là cái ly (áo PI-ÊU của TÂY móc vào cái LY rồi chạy trốn)", "reading": "ひょう", "vocab": [("剽窃", "ひょうせつ", "sự ăn cắp, sự đạo văn"), ("剽悍", "ひょうかん", "hung hãn, táo bạo")]},
    "瓢": {"viet": "BIỀU", "meaning_vi": "Cái bầu, lấy vỏ quả bầu chế ra đồ đựng rượu đựng nước, gọi là biều. Xem chữ hồ [瓠].", "meo": "", "reading": "ひさご ふくべ", "vocab": [("瓢箪", "ひょうたん", "bầu; bí ."), ("瓢虫", "てんとうむし", "con bọ rùa .")]},
    "凹": {"viet": "AO", "meaning_vi": "Lõm. Nguyễn Du [阮攸] : Ngạch đột diện ao [額凸面凹] (Long Thành cầm giả ca [龍城琴者歌]) Trán dô mặt gẫy.", "meo": "", "reading": "くぼ.む へこ.む ぼこ", "vocab": [("凹", "おう", "chỗ lõm"), ("凹み", "へこみ", "hình rập nổi")]},
    "凸": {"viet": "ĐỘT", "meaning_vi": "Lồi. Nguyễn Du [阮攸] : Ngạch đột diện ao [額凸面凹] (Long Thành cầm giả ca [龍城琴者歌]) Trán dô mặt gẫy.", "meo": "", "reading": "でこ", "vocab": [("お凸", "おでこ", "trán; trán dô"), ("両凸", "りょうとつ", "hai mặt lồi .")]},
    "蛍": {"viet": "HUỲNH", "meaning_vi": "Đom đóm,", "meo": "", "reading": "ほたる", "vocab": [("蛍", "ほたる", "con đom đóm ."), ("蛍光", "けいこう", "sự huỳnh quang; phát huỳnh quang")]},
    "蚕": {"viet": "TÀM", "meaning_vi": "Tục dùng như chữ tàm [蠶], nguyên là chữ điến là giống giun.", "meo": "", "reading": "かいこ こ", "vocab": [("蚕", "かいこ", "con tằm; tằm"), ("蚕児", "さんじ", "con tằm")]},
    "繭": {"viet": "KIỂN", "meaning_vi": "Cái kén tằm, tức là cái tổ của con tằm nó tụ nhả tơ ra để che mình nó.", "meo": "", "reading": "まゆ きぬ", "vocab": [("繭", "まゆ", "kén tằm ."), ("繭玉", "まゆだま", "tiền lì xì năm mới .")]},
    "濁": {"viet": "TRỌC, TRẠC", "meaning_vi": "Nước đục, đục. Như ô trọc [汙濁] đục bẩn.", "meo": "", "reading": "にご.る にご.す", "vocab": [("濁す", "にごす", "làm đục"), ("濁り", "にごり", "dấu phụ âm trong tiếng Nhật; sự không rõ ràng")]},
    "燭": {"viet": "CHÚC", "meaning_vi": "Đuốc, nến. Nguyễn Trãi [阮廌] : Duyên giang thiên lý chúc quang hồng [沿江千里燭光紅] (Thượng nguyên hỗ giá chu trung tác [上元扈駕舟中作]) Ven sông nghìn dặm, ánh đuốc đỏ rực.", "meo": "", "reading": "ともしび", "vocab": [("燭光", "しょっこう", "nến ."), ("燭台", "しょくだい", "cây đèn nến")]},
    "囚": {"viet": "TÙ", "meaning_vi": "Bỏ tù, bắt người có tội giam lại gọi là tù.", "meo": "", "reading": "とら.われる", "vocab": [("囚人", "しゅうじん", "tù"), ("免囚", "めんしゅう", "sự ra tù; cựu tù nhân .")]},
    "柵": {"viet": "SÁCH", "meaning_vi": "Hàng rào, cắm tre gỗ làm hàng rào để ngăn người đi gọi là sách.", "meo": "", "reading": "しがら.む しがらみ とりで やらい", "vocab": [("柵", "さく", "hàng rào cọc"), ("柵", "しがらみ", "bờ giậu")]},
    "珊": {"viet": "SAN", "meaning_vi": "San hô [珊瑚] một thứ động vật nhỏ ở trong bể kết lại, hình như cành cây, đẹp như ngọc, dùng làm chỏm mũ rất quý.", "meo": "", "reading": "センチ さんち", "vocab": [("珊瑚", "さんご", "san hô"), ("珊瑚礁", "さんごしょう", "bãi san hô")]},
    "倫": {"viet": "LUÂN", "meaning_vi": "Thường. Như luân lí [倫理] cái đạo thường người ta phải noi theo.", "meo": "", "reading": "リン", "vocab": [("倫", "りん", "bạn"), ("不倫", "ふりん", "bất luân; không còn luân thường đạo lý")]},
    "扁": {"viet": "BIẾN", "meaning_vi": "dẹt, phẳng, bộ phận", "meo": "Mái nhà (户) bị gập (艹) thành miếng dẹt.", "reading": "へん", "vocab": [("扁平", "へんぺい", "bẹt, dẹt"), ("偏見", "へんけん", "thiên kiến, thành kiến")]},
    "遍": {"viet": "BIẾN", "meaning_vi": "Khắp. Nguyễn Du [阮攸] : Khứ biến đông nam lộ [去遍東南路] (Chu phát [舟發]) Đi khắp đường đông nam.", "meo": "", "reading": "あまね.く", "vocab": [("一遍", "いっぺん", "một lần"), ("何遍", "なんべん", "bao nhiêu lần .")]},
    "偏": {"viet": "THIÊN", "meaning_vi": "Lệch, mếch, ở vào hai bên một cái gì gọi là thiên.", "meo": "", "reading": "かたよ.る", "vocab": [("偏", "へん", "mặt"), ("偏に", "ひとえに", "nghiêm túc")]},
    "煎": {"viet": "TIÊN, TIỄN", "meaning_vi": "Nấu, sắc, chất nước đem đun cho đặc gọi là tiên.", "meo": "", "reading": "せん.じる い.る に.る", "vocab": [("煎", "い", "thịt nướng"), ("煎る", "いる", "rang")]},
    "揃": {"viet": "TIỄN", "meaning_vi": "Hoàn tất; tương đương; đồng phục.", "meo": "", "reading": "そろ.える そろ.う そろ.い き.る", "vocab": [("揃い", "そろい", "bộ"), ("揃う", "そろう", "được thu thập; sẵn sàng; được sắp xếp một cách có trật tự")]},
    "愉": {"viet": "DU, THÂU", "meaning_vi": "Vui vẻ. Nét mặt hòa nhã vui vẻ gọi là du sắc [愉色].", "meo": "", "reading": "たの.しい たの.しむ", "vocab": [("愉快", "ゆかい", "hài lòng; thỏa mãn"), ("愉しい", "たのしい", "thú vị")]},
    "諭": {"viet": "DỤ", "meaning_vi": "Bảo, người trên bảo người dưới gọi là dụ. Như thượng dụ [上諭] dụ của vua.", "meo": "", "reading": "さと.す", "vocab": [("諭し", "さとし", "sự chỉ đạo"), ("諭す", "さとす", "dạy bảo; thuyết phục; huấn thị")]},
    "喩": {"viet": "DỤ", "meaning_vi": "metaphor", "meo": "", "reading": "たと.える さと.す", "vocab": [("喩え", "たとえ", "sự so sánh"), ("引喩", "いんゆ", "sự nói bóng gió")]},
    "癒": {"viet": "DŨ", "meaning_vi": "Ốm khỏi. Cũng viết là [愈] hay [瘉].", "meo": "", "reading": "い.える いや.す い.やす", "vocab": [("癒す", "いやす", "chữa khỏi"), ("癒合", "ゆごう", "sự dính kết")]},
    "霊": {"viet": "LINH", "meaning_vi": "Linh hồn", "meo": "", "reading": "たま", "vocab": [("霊", "れい", "linh hồn"), ("亡霊", "ぼうれい", "linh hồn đã chết; ma quỷ; vong linh")]},
    "譜": {"viet": "PHỔ, PHẢ", "meaning_vi": "Phả, sổ chép về nhân vật và chia rành thứ tự. Như gia phổ [家譜] phả chép thế thứ trong nhà họ.", "meo": "", "reading": "フ", "vocab": [("印譜", "いんぷ", "sự giàu có"), ("家譜", "いえふ", "phả hệ")]},
    "顕": {"viet": "HIỂN", "meaning_vi": "Hiển hách, hiển thị", "meo": "", "reading": "あきらか あらわ.れる", "vocab": [("顕在", "けんざい", "(+ up"), ("顕彰", "けんしょう", "sự khen thưởng; sự tuyên dương; khen thưởng; tuyên dương")]},
    "妄": {"viet": "VỌNG", "meaning_vi": "Sằng, càn. Như vọng ngữ [妄語] nói sằng, trái lại với chữ chân [真].", "meo": "", "reading": "みだ.りに", "vocab": [("妄信", "ぼうしん", "tính cả tin; tính nhẹ dạ ."), ("妄動", "もうどう", "sự nổi dậy")]},
    "盲": {"viet": "MANH", "meaning_vi": "Thanh manh, lòa.", "meo": "", "reading": "めくら", "vocab": [("盲", "めくら", "sự mù; người mù"), ("盲", "もう", "đui .")]},
    "網": {"viet": "VÕNG", "meaning_vi": "Cái lưới, cái chài. Tô Thức [蘇軾] : Cử võng đắc ngư, cự khẩu tế lân [舉網得魚, 巨口細鱗] (Hậu Xích Bích phú [後赤壁賦]) Cất lưới được cá, miệng to vẩy nhỏ.", "meo": "", "reading": "あみ", "vocab": [("網", "あみ", "chài"), ("天網", "てんもう", "lưới trời")]},
    "径": {"viet": "KÍNH", "meaning_vi": "Giản thể của chữ 徑", "meo": "", "reading": "みち こみち さしわたし ただちに", "vocab": [("内径", "ないけい", "Đường kính trong"), ("半径", "はんけい", "bán kính")]},
    "怪": {"viet": "QUÁI", "meaning_vi": "Lạ. Như quái sự [怪事] việc lạ.", "meo": "", "reading": "あや.しい あや.しむ", "vocab": [("怪", "かい", "điều huyền bí"), ("怪傑", "かいけつ", "sự giải quyết")]},
    "茎": {"viet": "HÀNH", "meaning_vi": "Giản thể của chữ 莖", "meo": "", "reading": "くき", "vocab": [("茎", "くき", "cọng; cuống"), ("一茎", "いちくき", "; cuống")]},
    "典": {"viet": "ĐIỂN", "meaning_vi": "Kinh điển, phép thường. Như điển hình [典刑] phép tắc. Tục viết là [典型].", "meo": "", "reading": "テン デン", "vocab": [("典", "てん", "bộ luật"), ("事典", "じてん", "bộ sách bách khoa")]},
    "膿": {"viet": "NÙNG", "meaning_vi": "Mủ.", "meo": "", "reading": "う.む うみ", "vocab": [("膿", "うみ", "mủ"), ("膿む", "うむ", "mưng mủ .")]},
    "艶": {"viet": "DIỄM", "meaning_vi": "Sắc người đẹp, tươi sáng. Cùng nghĩa với chữ diễm [豔].", "meo": "", "reading": "つや なま.めかしい あで.やか つや.めく なま.めく", "vocab": [("艶", "つや", "độ bóng; sự nhẵn bóng"), ("艶々", "つやつや", "bóng bảy")]},
    "乃": {"viet": "NÃI, ÁI", "meaning_vi": "Bèn, tiếng nói nối câu trên. Tô Thức [蘇軾] : Dư nãi nhiếp y nhi thướng [予乃攝衣而上] (Hậu Xích Bích phú [後赤壁賦]) Tôi bèn vén áo mà lên.", "meo": "", "reading": "の すなわ.ち なんじ", "vocab": [("乃", "の", "của"), ("乃父", "だいふ", "cha")]},
    "秀": {"viet": "TÚ", "meaning_vi": "ưu tú, xuất sắc", "meo": "Cây lúa (禾) mọc trên mái nhà (乃) thì thật là ưu tú.", "reading": "しゅう", "vocab": [("秀才", "しゅうさい", "tài năng, người có tài"), ("優秀", "ゆうしゅう", "ưu tú, xuất sắc")]},
    "透": {"viet": "THẤU", "meaning_vi": "Suốt qua. Như thấu minh [透明] ánh sáng suốt qua. Vì thế nên người nào tỏ rõ sự lý gọi là thấu triệt [透徹].", "meo": "", "reading": "す.く す.かす す.ける とう.る とう.す", "vocab": [("透き", "すき", "khoảng (thời gian"), ("透く", "すく", "hé; hở")]},
    "莫": {"viet": "MẠC, MỘ, MẠCH, BÁ", "meaning_vi": "Tuyệt không, chẳng ai không. Như mạc bất chấn cụ [莫不震懼] chẳng ai là chẳng sợ run.", "meo": "", "reading": "くれ なか.れ なし", "vocab": [("莫大", "ばくだい", "sự to lớn"), ("索莫", "さくばく", "tồi tàn")]},
    "模": {"viet": "MÔ", "meaning_vi": "Khuôn mẫu. Như mô phạm [模範] khuôn mẫu, chỉ ông thầy, mô dạng [模樣] hình dạng, dáng điệu, v.v.", "meo": "", "reading": "モ ボ", "vocab": [("模倣", "もほう", "mô phỏng"), ("模写", "もしゃ", "sao lại")]},
    "膜": {"viet": "MÔ, MẠC", "meaning_vi": "Màng, một thứ màng mỏng để ràng rịt tạng phủ và các cơ quan trong mình. Như nhãn mô [眼膜] màng mắt, nhĩ mô [耳膜] màng tai, v.v.", "meo": "", "reading": "マク", "vocab": [("膜", "まく", "màng ."), ("内膜", "ないまく", "Màng trong")]},
    "漠": {"viet": "MẠC", "meaning_vi": "Bãi sa mạc (bể cát). Như đại mạc chi trung [大漠之中] nơi xa mạc.", "meo": "", "reading": "バク", "vocab": [("広漠", "こうばく", "rộng lớn"), ("沙漠", "さばく", "công lao")]},
    "摸": {"viet": "MÔ", "meaning_vi": "sờ, mò, bắt chước", "meo": "Tay (扌) to (莫) mò trộm.", "reading": "さぐる", "vocab": [("摸索", "もさく", "mò mẫm, tìm tòi"), ("模倣", "もほう", "mô phỏng, bắt chước")]},
    "墓": {"viet": "MỘ", "meaning_vi": "Cái mả. Nguyễn Du [阮攸] : Tống triều cổ mộ kí Âu Dương [宋朝古墓記歐陽] (Âu Dương Văn Trung Công mộ [歐陽文忠公墓]) Ghi rõ mộ cổ của Âu Dương Tu đời nhà Tống.", "meo": "", "reading": "はか", "vocab": [("墓", "はか", "mả"), ("墓地", "はかち", "Nghĩa địa; bãi tha ma")]},
    "慕": {"viet": "MỘ", "meaning_vi": "Ngưỡng mộ, yêu mến", "meo": "Tim (心) chiều (莫) theo người mình mộ.", "reading": "したう", "vocab": [("慕う", "したう", "Ngưỡng mộ, yêu mến"), ("思慕", "しぼ", "Tưởng mộ, nhớ nhung")]},
    "幕": {"viet": "MẠC, MỘ, MÁN", "meaning_vi": "Cái màn che ở trên gọi là mạc. Trong quân phải giương màn lên để ở, nên chỗ quan tướng ở gọi là mạc phủ [幕府]. Các ban khách coi việc văn thư ở trong phủ gọi là mạc hữu [幕友], thường gọi tắt là mạc. Nay thường gọi các người coi việc tờ bồi giấy má ở trong nhà là mạc, là bởi nghĩa đó. Thường đọc là mộ", "meo": "", "reading": "とばり", "vocab": [("幕", "とばり", "màn; rèm ."), ("倒幕", "とうばく", "sự lật đổ chế độ Mạc phủ .")]},
    "其": {"viet": "KỲ", "meaning_vi": "kia, ấy, đó", "meo": "Bát úp ngược (八) bên trên cái bàn (radical bộ Kệ)", "reading": "そ", "vocab": [("其の", "その", "cái đó, cái ấy"), ("其処", "そこ", "chỗ đó, nơi đó")]},
    "棋": {"viet": "KÌ, KÍ", "meaning_vi": "Cờ, một thứ trò chơi, hai bên bày quân đánh nhau. Như thế đánh trận, ngày xưa gọi là tượng kì hí [象棋戲].", "meo": "", "reading": "ご", "vocab": [("棋士", "きし", "ngưòi chơi cờ chuyên nghiệp; cờ thủ"), ("将棋", "しょうぎ", "cờ bạc")]},
    "碁": {"viet": "KÌ", "meaning_vi": "Cùng nghĩa với chữ kì [棋].", "meo": "", "reading": "ゴ", "vocab": [("碁", "ご", "cờ gô"), ("囲碁", "いご", "cờ vây; cờ gô")]},
    "欺": {"viet": "KHI", "meaning_vi": "Dối lừa. Lừa mình, tự lừa dối mình gọi là tự khi [自欺].", "meo": "", "reading": "あざむ.く", "vocab": [("欺く", "あざむく", "đánh"), ("欺瞞", "ぎまん", "sự dối trá")]},
    "旗": {"viet": "KÌ", "meaning_vi": "Cờ, dùng vải hay lụa buộc lên cái cán để làm dấu hiệu gọi là kì. Như kì xí [旗幟] cờ xí.", "meo": "", "reading": "はた", "vocab": [("旗", "はた", "cờ; lá cờ ."), ("一旗", "ひとはた", "cây irit")]},
    "塞": {"viet": "TẮC, TÁI", "meaning_vi": "Lấp kín. Nguyễn Trãi [阮廌] : Kình du tắc hải, hải vi trì [鯨遊塞海海爲池] (Long Đại nham [龍袋岩]) Cá kình bơi lấp biển, biển thành ao.", "meo": "", "reading": "ふさ.ぐ とりで み.ちる", "vocab": [("塞ぐ", "ふさぐ", "bế tắc"), ("充塞", "じゅうそく", "nút (chậu sứ rửa mặt")]},
    "甚": {"viet": "THẬM", "meaning_vi": "rất, lắm, quá", "meo": "Nhìn giống chữ NHẤT đang đội MŨ LỚN để đi THĂM ai đó (THẬM).", "reading": "じん", "vocab": [("甚大", "じんだい", "to lớn, nghiêm trọng"), ("甚だ", "はなはだ", "rất, lắm, quá (trang trọng)")]},
    "堪": {"viet": "KHAM", "meaning_vi": "Chịu được. Như bất kham [不堪] chẳng chịu được.", "meo": "", "reading": "た.える たま.る こら.える こた.える", "vocab": [("不堪", "ふかん", "sự thiếu khả năng"), ("堪忍", "かんにん", "sự dễ dàng khoan dung")]},
    "勘": {"viet": "KHÁM", "meaning_vi": "So sánh, định lại. Như xét lại văn tự để sửa chỗ nhầm gọi là hiệu khám [校勘].", "meo": "", "reading": "カン", "vocab": [("勘", "かん", "trực giác; giác quan thứ sáu; cảm tính; linh cảm"), ("勘例", "かんれい", "phong tục")]},
    "炊": {"viet": "XUY, XÚY", "meaning_vi": "Thổi nấu, lấy lửa đun cho chín đồ ăn gọi là xuy. Nguyễn Du [阮攸 ] : Mại ca khất tiền cung thần xuy [賣歌乞錢供晨炊] (Thái Bình mại ca giả [太平賣歌者]) Hát dạo xin tiền nấu ăn.", "meo": "", "reading": "た.く -だ.き", "vocab": [("炊く", "たく", "đun sôi; nấu sôi; nấu"), ("炊事", "すいじ", "việc bếp núc; nghệ thuật nấu nướng .")]},
    "諮": {"viet": "TI, TƯ", "meaning_vi": "Mưu, hỏi. Ta quen đọc là chữ tư. Tư tuân dân ý [諮詢民意] trưng cầu dân ý.", "meo": "", "reading": "はか.る", "vocab": [("諮る", "はかる", "hỏi ý"), ("諮問", "しもん", "yêu cầu; tư vấn; cố vấn .")]},
    "恣": {"viet": "TỨ, THƯ", "meaning_vi": "Phóng túng, tự ý làm láo không kiêng nể gì gọi là tứ.", "meo": "", "reading": "ほしいまま", "vocab": [("恣意", "しい", "tính ích kỷ"), ("放恣", "ほうし", "phóng túng")]},
    "羨": {"viet": "TIỆN, TIỂN", "meaning_vi": "Tham muốn, lòng ham thích cái gì gọi là tiện. Nguyễn Du [阮攸] : Chí kim thùy phục tiện Trương Khiên [至今誰復羨張騫] (Hoàng Hà [黄河]) Đến nay còn ai muốn làm như Trương Khiên (Thời Hán, Trương Khiên theo sông Hoàng Hà 黃河 đi thuyết phục nhiều nước miền Tây Bắc Trung Quốc).", "meo": "", "reading": "うらや.む あまり", "vocab": [("羨む", "うらやむ", "đố"), ("羨望", "せんぼう", "sự thèm muốn")]},
    "款": {"viet": "KHOẢN", "meaning_vi": "Thành thực. Như khổn khoản [悃款] khẩn khoản, tả cái chí thuần nhất, thành thực.", "meo": "", "reading": "カン", "vocab": [("借款", "しゃっかん", "khoản vay ."), ("定款", "ていかん", "điều lệ .")]},
    "逢": {"viet": "PHÙNG, BỒNG", "meaning_vi": "Gặp. Hai bên gặp nhau gọi là phùng.", "meo": "", "reading": "あ.う むか.える", "vocab": [("逢う", "あう", "gặp gỡ; hợp; gặp"), ("逢引", "あいびき", "Hẹn hò lén lút của trai gái; mật hội; hội kín; họp kín")]},
    "縫": {"viet": "PHÙNG, PHÚNG", "meaning_vi": "May áo.", "meo": "", "reading": "ぬ.う", "vocab": [("縫い", "ぬい", "Việc khâu vá ."), ("縫う", "ぬう", "khâu")]},
    "峰": {"viet": "PHONG", "meaning_vi": "Ngọn núi.", "meo": "", "reading": "みね ね", "vocab": [("峰", "みね", "chóp; ngọn; đỉnh ."), ("主峰", "しゅほう", "kỹ xảo")]},
    "蜂": {"viet": "PHONG", "meaning_vi": "Con ong.", "meo": "", "reading": "はち", "vocab": [("蜂", "はち", "ong"), ("蜂巣", "はちす", "tổ ong")]},
    "且": {"viet": "THẢ", "meaning_vi": "Vả lại, hơn nữa", "meo": "Vừa thấy chữ 'Nhất' ở trên, vừa thấy chữ 'Nguyệt' ở dưới thì 'Lại' buồn, 'Lại' suy tư.", "reading": "かつ", "vocab": [("且つ", "かつ", "Hơn nữa, đồng thời"), ("且", "しばらく", "Một lát, chốc nữa")]},
    "粗": {"viet": "THÔ", "meaning_vi": "Vầng to. Như thô tế [粗細] vầng to nhỏ, dùng để nói về chu vi to hay nhỏ.", "meo": "", "reading": "あら.い あら-", "vocab": [("粗", "ほぼ", "thiếu sót"), ("粗い", "あらい", "thô; cục mịch; gồ ghề; khấp khiểng; lổn nhổn")]},
    "租": {"viet": "TÔ", "meaning_vi": "Thuế ruộng, bán ruộng cho người cấy thuê cũng gọi là điền tô [田租].", "meo": "", "reading": "ソ", "vocab": [("租借", "そしゃく", "sự cho thuê"), ("地租", "ちそ", "điền tô")]},
    "狙": {"viet": "THƯ", "meaning_vi": "Một giống như con vượn, tính rất giảo quyệt.", "meo": "", "reading": "ねら.う ねら.い", "vocab": [("狙い", "ねらい", "mục đích"), ("狙う", "ねらう", "nhắm vào; nhằm mục đích")]},
    "阻": {"viet": "TRỞ", "meaning_vi": "Hiểm trở. Chỗ núi hiểm hóc gọi là hiểm [險], chỗ nước nguy hiểm gọi là trở [阻].", "meo": "", "reading": "はば.む", "vocab": [("阻む", "はばむ", "cản trở; ngăn cản"), ("阻喪", "そそう", "sự buồn nản")]},
    "宜": {"viet": "NGHI", "meaning_vi": "thích hợp, nên", "meo": "Mái nhà (宀) có Một (一) Miếng thịt (且) là phù hợp (宜) để cúng tổ tiên.", "reading": "ぎ", "vocab": [("宜しい", "よろしい", "tốt, được, chấp nhận được"), ("便宜", "べんぎ", "tiện nghi, thuận lợi")]},
    "誼": {"viet": "NGHỊ", "meaning_vi": "Cũng như chữ nghĩa [義].", "meo": "", "reading": "よしみ よい", "vocab": [("誼", "よしみ", "tình bạn"), ("誼み", "よしみみ", "tình bạn")]},
    "拘": {"viet": "CÂU, CÙ", "meaning_vi": "Bắt. Như bị câu [被拘] bị bắt.", "meo": "", "reading": "かか.わる", "vocab": [("拘引", "こういん", "sự bắt giữ"), ("拘らず", "かかわらず", "không chú ý đến ; không quan tâm")]},
    "駒": {"viet": "CÂU", "meaning_vi": "Ngựa hai tuổi gọi là câu. Phàm ngựa còn non còn khoẻ đều gọi là câu cả. Vì thế khen các con em có tài khí hơn người gọi là thiên lí câu [千里駒].", "meo": "", "reading": "こま", "vocab": [("駒", "こま", "con cù"), ("一駒", "いちこま", "nơi xảy ra")]},
    "驚": {"viet": "KINH", "meaning_vi": "Ngựa sợ hãi.", "meo": "", "reading": "おどろ.く おどろ.かす", "vocab": [("驚き", "おどろき", "sự ngạc nhiên"), ("驚く", "おどろく", "giật mình")]},
    "崖": {"viet": "NHAI", "meaning_vi": "Ven núi. Cũng như chữ nhai [厓]. Nguyễn Du [阮攸] : Hồi đầu dĩ cách vạn trùng nhai [回頭已隔萬重崖] (Vọng Quan Âm miếu [望觀音廟]) Quay đầu lại đã cách núi muôn trùng.", "meo": "", "reading": "がけ きし はて", "vocab": [("崖", "がけ", "vách đá dốc đứng"), ("崎崖", "きがい", "Độ dốc của ngọn núi .")]},
    "涯": {"viet": "NHAI", "meaning_vi": "Bờ bến. Cái gì vô cùng vô tận gọi là vô nhai [無涯]. Nguyễn Trãi [阮廌] : Bồng Lai Nhược Thủy yểu vô nhai [篷萊弱水杳無涯] (Họa hữu nhân yên hà ngụ hứng [和友人煙寓興]) Non Bồng nước Nhược mờ mịt không bờ bến.", "meo": "", "reading": "はて", "vocab": [("天涯", "てんがい", "đường chân trời"), ("生涯", "しょうがい", "sinh nhai; cuộc đời .")]},
    "佳": {"viet": "GIAI", "meaning_vi": "Tốt, quý. Nguyễn Trãi [阮薦] : Giai khách tương phùng nhật bão cầm [佳客相逢日抱琴] (Đề Trình xử sĩ Vân oa đồ [題程處士雲窩圖]) Khách quý gặp nhau, ngày ngày ôm đàn gảy.", "meo": "", "reading": "カ", "vocab": [("佳", "けい", "đẹp; hay"), ("佳人", "かじん", "người phụ nữ đẹp; hồng nhan; giai nhân; bóng hồng")]},
    "罫": {"viet": "QUẢI", "meaning_vi": "Trở ngại.", "meo": "", "reading": "ケイ カイ ケ", "vocab": [("罫線", "けいせん", "phép tắc"), ("罫線表", "けいせんひょう", "bản đồ đi biển")]},
    "鮭": {"viet": "KHUÊ, HÀI", "meaning_vi": "Một tên riêng của con lợn bể, cá hồi.", "meo": "", "reading": "さけ しゃけ ふぐ", "vocab": [("鮭", "さけ", "cá hồi ."), ("鮭漁", "さけりょう", "sự câu cá hồi .")]},
    "窪": {"viet": "OA", "meaning_vi": "Chỗ trũng.", "meo": "", "reading": "くぼ.む くぼ.み くぼ.まる くぼ", "vocab": [("窪み", "くぼみ", "lỗ; hốc; chỗ lõm"), ("窪む", "くぼむ", "thùng rửa bát")]},
    "桂": {"viet": "QUẾ", "meaning_vi": "Cây quế, dùng để làm thuốc.", "meo": "", "reading": "かつら", "vocab": [("桂", "かつら", "bộ tóc gi"), ("月桂", "げっけい", "cây nguyệt quế")]},
    "橘": {"viet": "QUẤT", "meaning_vi": "Cây quất (cây quýt).", "meo": "", "reading": "たちばな", "vocab": []},
    "胴": {"viet": "ĐỖNG", "meaning_vi": "Cái thân người, từ cổ xuống đến bẹn, trừ chân tay ra, gọi là đỗng.", "meo": "", "reading": "ドウ", "vocab": [("胴", "どう", "cơ thể ."), ("胴中", "どうなか", "thân (cây")]},
    "桐": {"viet": "ĐỒNG", "meaning_vi": "Cây đồng (cây vông); một thứ gỗ dùng để đóng đàn.", "meo": "", "reading": "きり", "vocab": [("梧桐", "あおぎり", "Cây ngô đồng ."), ("桐油", "とうゆ", "dầu tung")]},
    "駆": {"viet": "KHU", "meaning_vi": "Khu trục hạm.", "meo": "", "reading": "か.ける か.る", "vocab": [("駆る", "かる", "bị... chi phối"), ("先駆", "せんく", "điềm báo trước; người đến báo trước; người tiên phong; người dẫn đường")]},
    "枢": {"viet": "XU", "meaning_vi": "Giản thể của chữ 樞", "meo": "", "reading": "とぼそ からくり", "vocab": [("中枢", "ちゅうすう", "trung khu; trung tâm"), ("枢密", "すうみつ", "bí mật quốc gia .")]},
    "鴎": {"viet": "", "meaning_vi": "seagull", "meo": "", "reading": "かもめ", "vocab": []},
    "愚": {"viet": "NGU", "meaning_vi": "Ngu dốt. Như ngu si [愚癡] dốt nát mê muội.", "meo": "", "reading": "おろ.か", "vocab": [("愚", "ぐ", "dại dột"), ("愚か", "おろか", "ngu ngốc; ngớ ngẩn")]},
    "遇": {"viet": "NGỘ", "meaning_vi": "Gặp, gặp nhau gọi là ngộ. Như hội ngộ [會遇] gặp gỡ.", "meo": "", "reading": "あ.う", "vocab": [("遇う", "あう", "cuộc gặp gỡ (của những người đi săn ở một nơi đã hẹn trước"), ("不遇", "ふぐう", "Vận rủi .")]},
    "萬": {"viet": "VẠN, MẶC", "meaning_vi": "Muôn, mười nghìn là một vạn [萬].", "meo": "", "reading": "よろず", "vocab": [("萬", "よろず", "tất cả"), ("萬釣り", "よろずづり", "sự thủ dâm")]},
    "寓": {"viet": "NGỤ", "meaning_vi": "Nhờ. Như ngụ cư [寓居] ở nhờ.", "meo": "", "reading": "ぐう.する かこつ.ける よ.せる よ.る かりずまい", "vocab": [("寓意", "ぐうい", "chủ nghĩa tượng trưng"), ("寓言", "ぐうげん", "phúng dụ")]},
    "廉": {"viet": "LIÊM", "meaning_vi": "Góc nhà, ở bên bệ thềm bước lên gọi là đường liêm [堂廉]. Như đường cao liêm viễn [堂高廉遠] nhà cao góc bệ xa, ý nói nhà vua cao xa lắm.", "meo": "", "reading": "レン", "vocab": [("一廉", "ひとかど", "sự cao hơn"), ("低廉", "ていれん", "rẻ")]},
    "簾": {"viet": "LIÊM", "meaning_vi": "Bức rèm, cái mành mành. Ngày xưa vua còn bé thì mẹ vua thì mẹ vua buông mành sử việc triều chính gọi là thùy liêm [垂簾], vua lớn lên, giao trả lại việc triều chính gọi là triệt liêm [撤簾].", "meo": "", "reading": "すだれ す", "vocab": [("暖簾", "のれん", "tấm rèm che trước cửa; danh tiếng của cửa hàng"), ("縄暖簾", "なわのれん", "rèm dây thừng")]},
    "鎌": {"viet": "LIÊM", "meaning_vi": "Cũng như chữ liêm [鐮].", "meo": "", "reading": "かま", "vocab": [("鎌", "かま", "liềm ."), ("大鎌", "おおがま", "cái hái hớt cỏ")]},
    "謙": {"viet": "KHIÊM, KHIỆM", "meaning_vi": "Nhún nhường, tự nhún nhường không dám khoe gọi là khiêm. Như khiêm nhượng [謙讓] nhún nhường.", "meo": "", "reading": "ケン", "vocab": [("恭謙", "きょうけん", "khiêm tốn; nhún nhường"), ("謙称", "けんしょう", "sự thẩm tra")]},
    "亥": {"viet": "HỢI", "meaning_vi": "Chi Hợi, một chi cuối cùng trong mười hai chi. Từ chín giờ đến mười một đêm gọi là giờ Hợi.", "meo": "", "reading": "い", "vocab": [("丁亥", "ていがい", "Đinh Hợi .")]},
    "咳": {"viet": "KHÁI", "meaning_vi": "Ho (ho không có đờm). Cũng như chữ khái [欬].", "meo": "", "reading": "せ.く しわぶ.く せき しわぶき", "vocab": [("咳", "せき", "bệnh ho"), ("咳く", "せく", "chứng ho; sự ho; tiếng ho")]},
    "該": {"viet": "CAI", "meaning_vi": "Cai quát đủ, nghĩa là bao quát hết thẩy. Như tường cai [詳該] tường tận.", "meo": "", "reading": "ガイ", "vocab": [("該博", "がいはく", "sự sâu"), ("当該", "とうがい", "thích hợp; phù hợp")]},
    "骸": {"viet": "HÀI", "meaning_vi": "Xương đùi.", "meo": "", "reading": "むくろ", "vocab": [("亡骸", "なきがら", "thi thể"), ("屍骸", "しがい", "thân thể")]},
    "劾": {"viet": "HẶC", "meaning_vi": "Hặc. Như sam hặc [参劾] bàn hặc (dự vào việc hặc tội người khác). Làm quan có lỗi tự thú tội mình gọi là tự hặc [自劾].", "meo": "", "reading": "ガイ", "vocab": [("弾劾", "だんがい", "sự đàn hặc; sự chỉ trích; sự buộc tội")]},
    "覆": {"viet": "PHÚC, PHÚ", "meaning_vi": "Lật lại. Kẻ nào hay giở giáo gọi là kẻ phản phúc vô thường [反覆無常]. Nguyễn Trãi [阮廌] : Phúc chu thủy tín dân do thủy [覆舟始信民猶水] (Quan hải [關海]) Thuyền lật mới tin dân như nước. Ý nói nhà cầm vận nước cần được lòng dân ủng hộ.", "meo": "", "reading": "おお.う くつがえ.す くつがえ.る", "vocab": [("覆い", "おおい", "áo khoác"), ("覆う", "おおう", "gói; bọc; che đậy; che giấu; bao phủ")]},
    "履": {"viet": "LÍ", "meaning_vi": "Giầy da, giầy đi đóng bằng da gọi là lí. Nguyễn Du [阮攸] : Phân hương mại lí khổ đinh ninh [分香賣履苦叮嚀] (Đồng Tước đài [銅雀臺]) Chia hương, bán giày, khổ tâm dặn dò.", "meo": "", "reading": "は.く", "vocab": [("履く", "はく", "đi (giày"), ("履歴", "りれき", "lịch sử; dữ kiện")]},
    "趣": {"viet": "THÚ, XÚC", "meaning_vi": "Rảo tới, đi mau tới chỗ đã định. Như thú lợi [趣利] nhanh chân kiếm lời.", "meo": "", "reading": "おもむき おもむ.く", "vocab": [("趣", "おもむき", "dáng vẻ; cảnh tượng; cảm giác; ấn tượng"), ("趣き", "おもむき", "dáng vẻ; cảnh tượng; cảm giác")]},
    "娶": {"viet": "THÚ", "meaning_vi": "Lấy vợ.", "meo": "", "reading": "めと.る めあわ.せる", "vocab": [("娶る", "めとる", "đẹp duyên .")]},
    "撮": {"viet": "TOÁT", "meaning_vi": "Dúm, phép đong ngày xưa cứ đếm 256 hạt thóc gọi là một toát.", "meo": "", "reading": "と.る つま.む -ど.り", "vocab": [("撮む", "つまむ", "cái vấu"), ("撮る", "とる", "chụp (ảnh); làm (phim) .")]},
    "偲": {"viet": "TƯ", "meaning_vi": "tưởng nhớ, hồi tưởng", "meo": "Người (亻) phương Tây (西) có một ít (、) năng lực (力) siêu nhiên để tưởng nhớ quá khứ.", "reading": "しのぶ", "vocab": [("偲ぶ", "しのぶ", "tưởng nhớ"), ("追偲", "ついし", "tưởng niệm, truy niệm")]},
    "穂": {"viet": "TUỆ", "meaning_vi": "Tai.", "meo": "", "reading": "ほ", "vocab": [("穂", "ほ", "bông (loại lúa"), ("穂先", "ほさき", "bông; nụ")]},
    "塁": {"viet": "LŨY", "meaning_vi": "Thành lũy", "meo": "", "reading": "とりで", "vocab": [("塁", "るい", "điều lo lắng"), ("一塁", "いちるい", "điểm đầu tiên trong bốn điểm phải được chạm bóng")]},
    "卑": {"viet": "TI", "meaning_vi": "Thấp.", "meo": "", "reading": "いや.しい いや.しむ いや.しめる", "vocab": [("下卑", "げび", "thông thường"), ("卑下", "ひげ", "sự tự hạ mình")]},
    "碑": {"viet": "BI", "meaning_vi": "Bia. Nguyễn Du [阮攸] : Thiên thu bi kiệt hiển tam liệt [千秋碑碣顯三烈] (Tam liệt miếu [三烈廟]) Bia kệ nghìn năm tôn thờ ba người tiết liệt.", "meo": "", "reading": "いしぶみ", "vocab": [("口碑", "こうひ", "truyện cổ tích"), ("墓碑", "ぼひ", "bia mộ; mộ chí .")]},
    "苗": {"viet": "MIÊU", "meaning_vi": "Lúa non, lúa mới cấy chưa tốt.", "meo": "", "reading": "なえ なわ-", "vocab": [("苗", "なえ", "cây con"), ("苗代", "なわしろ", "ruộng mạ")]},
    "描": {"viet": "MIÊU", "meaning_vi": "Phỏng vẽ, nghĩa là trông bức vẽ nào hay chữ nào mà vẽ phỏng ra, viết phỏng ra cho giống vậy. Như miêu tả [描寫] dùng nét vẽ hoặc lời văn mà vẽ lại, viết lại những điều mình thấy.", "meo": "", "reading": "えが.く か.く", "vocab": [("描く", "えがく", "vẽ; tô vẽ; mô tả; miêu tả"), ("描く", "かく", "chấm")]},
    "錨": {"viet": "MIÊU", "meaning_vi": "Cái mỏ neo để cắm thuyền tàu.", "meo": "", "reading": "いかり", "vocab": [("錨", "いかり", "cái neo; mỏ neo"), ("投錨", "とうびょう", "sự thả neo; sự hạ neo")]},
    "縛": {"viet": "PHƯỢC, PHỌC", "meaning_vi": "Trói buộc. Như tựu phược [就縛] bắt trói, chịu trói. Đỗ Phủ [杜甫] : Tiểu nô phược kê hướng thị mại [小奴縛雞向市賣] (Phược kê hành [縛雞行]) Đứa đầy tớ nhỏ trói gà đem ra chợ bán.", "meo": "", "reading": "しば.る", "vocab": [("縛る", "しばる", "buộc; trói; băng bó"), ("呪縛", "じゅばく", "sự nguyền rủa")]},
    "瞭": {"viet": "LIỆU", "meaning_vi": "Mắt sáng, mắt trong sáng sủa.", "meo": "", "reading": "あきらか", "vocab": [("明瞭", "めいりょう", "rõ ràng; sáng sủa"), ("瞭然", "りょうぜん", "rõ ràng")]},
    "隙": {"viet": "KHÍCH", "meaning_vi": "Cái lỗ hổng trên tường trên vách. Như sách Mạnh Tử [孟子] nói toàn huyệt khích tương khuy [鑽穴隙相窺] chọc lỗ tường cùng nhòm.", "meo": "", "reading": "すき す.く す.かす ひま", "vocab": [("隙", "すき", "cơ hội; dịp; khe hở; kẽ hở; khe hở trong lập luận ."), ("寸隙", "すんげき", "bài thơ trào phúng")]},
    "竜": {"viet": "LONG", "meaning_vi": "Như chữ long [龍].", "meo": "", "reading": "たつ いせ", "vocab": [("竜", "りゅう", "rồng"), ("土竜", "むぐらもち", "đê chắn sóng")]},
    "篭": {"viet": "", "meaning_vi": "seclude oneself, cage, coop, implied", "meo": "", "reading": "かご こ.める こも.る こ.む", "vocab": [("篭", "かご", "cái giỏ; giỏ; cái lồng; lồng; cái rổ; rổ; cái hom; hom"), ("揺篭", "ゆらかご", "cái nôi")]},
    "俺": {"viet": "YÊM", "meaning_vi": "Ta đây.", "meo": "", "reading": "おれ われ", "vocab": [("俺", "おれ", "tao; tôi"), ("俺等", "おれとう", "chúng tôi")]},
    "鬼": {"viet": "QUỶ", "meaning_vi": "Ma, người chết gọi là quỷ. Như ngạ quỷ [餓鬼] ma đói. Tam Quốc Diễn Nghĩa [三國演義] : Dạ dạ chỉ văn đắc thủy biên quỷ khốc thần hào [夜夜只聞得水邊鬼哭神號] (Hồi 91) Đêm đêm chỉ nghe bên sông ma khóc thần gào.", "meo": "", "reading": "おに おに-", "vocab": [("鬼", "おに", "con quỉ"), ("鬼女", "きじょ", "nữ quỷ; quỷ cái .")]},
    "醜": {"viet": "XÚ, SỬU", "meaning_vi": "Xấu. Tục dùng làm một tiếng để mắng nhiếc người.", "meo": "", "reading": "みにく.い しこ", "vocab": [("醜い", "みにくい", "xấu xí ."), ("醜女", "しこめ", "người phụ nữ chất phác; người phụ nữ giản dị")]},
    "塊": {"viet": "KHỐI", "meaning_vi": "Hòn, phàm vật gì tích dần lại thành từng hòn gọi là khối. Như băng khối [冰塊] tảng băng.", "meo": "", "reading": "かたまり つちくれ", "vocab": [("塊", "かたまり", "cục; tảng; miếng"), ("一塊", "いちかたまり", "nhóm")]},
    "椿": {"viet": "XUÂN, THUNG", "meaning_vi": "Cây xuân. Ông Trang Tử [莊子] nói đời xưa có cây xuân lớn, lấy tám nghìn năm làm một mùa xuân, tám nghìn năm làm một mùa thu. Vì thế, người ta hay dùng chữ xuân để chúc thọ. Nay ta gọi cha là xuân đình [椿庭] cũng theo ý ấy. $ Tục đọc là chữ thung.", "meo": "", "reading": "つばき", "vocab": [("椿", "つばき", "Cây hoa trà ."), ("椿油", "つばきあぶら", "Dầu hoa trà .")]},
    "奏": {"viet": "TẤU", "meaning_vi": "Tâu, kẻ dưới trình bầy với người trên gọi là tấu.", "meo": "", "reading": "かな.でる", "vocab": [("伏奏", "ふくそう", "sự hội tụ"), ("伝奏", "でんそう", "sự tấu truyền .")]},
    "泰": {"viet": "THÁI", "meaning_vi": "To lớn, cùng nghĩa với chữ thái [太].", "meo": "", "reading": "タイ", "vocab": [("安泰", "あんたい", "hòa bình; bằng phẳng; ổn"), ("泰平", "たいへい", "sự thái bình; sự yên bình; sự thanh bình")]},
    "奉": {"viet": "PHỤNG, BỔNG", "meaning_vi": "Vâng, kính vâng mệnh ý của người trên gọi là phụng.", "meo": "", "reading": "たてまつ.る まつ.る ほう.ずる", "vocab": [("奉る", "たてまつる", "mời; biếu; tôn trọng"), ("奉仕", "ほうし", "sự phục vụ; sự lao động .")]},
    "俸": {"viet": "BỔNG", "meaning_vi": "Bổng lộc.", "meo": "", "reading": "ホウ", "vocab": [("俸", "ほう", "lương bổng; bổng lộc ."), ("加俸", "かほう", "pháo")]},
    "捧": {"viet": "PHỦNG", "meaning_vi": "Bưng. Lý Hoa [李華] : Đề huề phủng phụ [提攜捧負] (Điếu cổ chiến trường văn [弔古戰場文]) Dắt díu nâng đỡ.", "meo": "", "reading": "ささ.げる", "vocab": [("捧呈", "ほうてい", "sự cống hiến"), ("捧げる", "ささげる", "giơ cao; giương lên; cống hiến; trình lên; đệ lên")]},
    "擦": {"viet": "SÁT", "meaning_vi": "Xoa, xát.", "meo": "", "reading": "す.る す.れる -ず.れ こす.る こす.れる", "vocab": [("擦る", "こする", "chà xát; lau; chùi"), ("擦る", "する", "cọ xát; chà xát; xát .")]},
    "柑": {"viet": "CAM", "meaning_vi": "Cây cam.", "meo": "", "reading": "コン カン", "vocab": [("蜜柑", "みかん", "quýt; quả quýt ."), ("金柑", "きんかん", "quả quất vàng")]},
    "某": {"viet": "MỖ", "meaning_vi": "Mỗ, dùng làm tiếng đệm. Như mỗ ông [某翁] ông mỗ, mỗ sự [某事] việc mỗ, v.v.", "meo": "", "reading": "それがし なにがし", "vocab": [("某", "ぼう", "nào đó"), ("某々", "ぼう々", "ông  này")]},
    "謀": {"viet": "MƯU", "meaning_vi": "Toan tính, toan tính trước rồi mới làm gọi là mưu. Như tham mưu [參謀] cùng dự vào việc mưu toan ấy, mưu sinh [謀生] toan mưu sự sinh nhai, nay gọi sự gặp mặt nhau là mưu diện [謀面] nghĩa là mưu toan cho được gặp mặt nhau một lần vậy.", "meo": "", "reading": "はか.る たばか.る はかりごと", "vocab": [("謀", "はかりごと", "mưu trí ."), ("謀る", "はかる", "lừa; tính kế")]},
    "媒": {"viet": "MÔI", "meaning_vi": "Mối, mưu cho hai họ kết dâu gia với nhau gọi là môi.", "meo": "", "reading": "なこうど", "vocab": [("媒介", "ばいかい", "môi giới; sự trung gian ."), ("媒体", "ばいたい", "người trung gian")]},
    "煤": {"viet": "MÔI", "meaning_vi": "Than mỏ, than đá. Cây cối đổ nát bị đất đè lên, lâu ngày đông lại thành ra than rắn như đá, sức lửa rất mạnh gọi là môi.", "meo": "", "reading": "すす", "vocab": [("煤", "すす", "mồ hóng"), ("煤煙", "ばいえん", "bồ hóng .")]},
    "嵌": {"viet": "KHẢM", "meaning_vi": "Hõm vào.", "meo": "", "reading": "は.める は.まる あな", "vocab": [("嵌る", "はまる", "(từ cổ"), ("嵌入", "かんにゅう", "đổ")]},
    "凶": {"viet": "HUNG", "meaning_vi": "Ác. Nhưng hung bạo [凶暴] ác dữ.", "meo": "", "reading": "キョウ", "vocab": [("凶", "きょう", "xấu"), ("凶事", "きょうじ", "tai hoạ")]},
    "圏": {"viet": "QUYỂN, KHUYÊN", "meaning_vi": "Một dạng của chữ [圈].", "meo": "", "reading": "かこ.い", "vocab": [("圏", "けん", "hình cầu"), ("圏内", "けんない", "trong khu vực; trong phạm vi")]},
    "倦": {"viet": "QUYỆN", "meaning_vi": "Mỏi mệt. Nguyễn Du [阮攸] : Đồ trường tê quyện mã [途長嘶倦馬] (Hà Nam đạo trung khốc thử [河南道中酷暑]) Đường dài, ngựa mệt hí vang.", "meo": "", "reading": "あき.る あぐ.む あぐ.ねる う.む つか.れる", "vocab": [("倦厭", "けんえん", "sự mệt mỏi"), ("倦怠", "けんたい", "sự mệt mỏi; sự chán chường; mệt mỏi; chán chường")]},
    "捲": {"viet": "QUYỂN, QUYỀN", "meaning_vi": "Cuốn, cũng như chữ quyển [卷]. Tịch quyển [席捲] cuốn tất (bao quát tất cả). Tô Thức [蘇軾] : Quyển khởi thiên đôi tuyết [捲起千堆雪] (Niệm nô kiều [念奴嬌]) Cuốn lôi ngàn đống tuyết. Tình sử [情史] : Trùng liêm bất quyển nhật hôn hoàng [重簾不捲日昏黃 ] Đôi tầng rèm rủ, bóng dương tà.", "meo": "", "reading": "ま.く ま.くる まく.る めく.る まく.れる", "vocab": [("捲く", "まく", "lên dây ."), ("捲る", "まくる", "xắn lên; vấn lên; quấn lên")]},
    "拳": {"viet": "QUYỀN", "meaning_vi": "Nắm tay lại.", "meo": "", "reading": "こぶし", "vocab": [("拳", "こぶし", "nắm tay; quả đấm"), ("拳固", "げんこ", "nắm tay; quả đấm")]},
    "藤": {"viet": "ĐẰNG", "meaning_vi": "Bụi cây quấn quít, loài thực vật thân cây mọc từng bụi.", "meo": "", "reading": "ふじ", "vocab": [("藤", "ふじ", "<THựC> cây đậu tía"), ("藤本", "とうほん", "dây leo .")]},
    "謄": {"viet": "ĐẰNG", "meaning_vi": "Sao, chép, trông bản kia sao ra bản khác cho rõ ràng. Như đằng hoàng [騰黃] lấy giấy vàng viết tờ chiếu cho rõ ràng.", "meo": "", "reading": "トウ", "vocab": [("謄写", "とうしゃ", "sự sao chép; sự sao lại ."), ("謄本", "とうほん", "mẫu; bản .")]},
    "騰": {"viet": "ĐẰNG", "meaning_vi": "Ngựa chạy mau. Như vạn mã bôn đằng [萬馬奔騰] muôn ngựa rong ruổi.", "meo": "", "reading": "あが.る のぼ.る", "vocab": [("上騰", "うえあが", "sự tiến lên"), ("騰勢", "とうせい", "Khuynh hướng đi lên .")]},
    "渓": {"viet": "KHÊ", "meaning_vi": "Thung lũng, khê cốc", "meo": "", "reading": "たに たにがわ", "vocab": [("渓流", "けいりゅう", "suối nước nguồn; mạch nước từ núi chảy ra"), ("渓谷", "けいこく", "đèo ải")]},
    "扶": {"viet": "PHÙ", "meaning_vi": "Giúp đỡ.", "meo": "", "reading": "たす.ける", "vocab": [("扶助", "ふじょ", "sự giúp đỡ; sự nâng đỡ; sự trợ giúp ."), ("家扶", "かふ", "người quản lý")]},
    "楷": {"viet": "GIAI, KHẢI", "meaning_vi": "Cây giai.", "meo": "", "reading": "カイ", "vocab": [("楷書", "かいしょ", "sự viết theo lối chân phương; lối chân phương")]},
    "諧": {"viet": "HÀI", "meaning_vi": "Hòa hợp. Như âm điệu ăn nhịp nhau gọi là hài thanh [諧聲], mua hàng ngã giá rồi gọi là hài giá [諧價].", "meo": "", "reading": "かな.う やわ.らぐ", "vocab": [("諧調", "かいちょう", "giai điệu"), ("諧謔", "かいぎゃく", "lời nói đùa")]},
    "昆": {"viet": "CÔN", "meaning_vi": "Con nối. Như hậu côn [後昆] đàn sau.", "meo": "", "reading": "コン", "vocab": [("昆布", "こぶ", "tảo bẹ"), ("昆布", "こんぶ", "tảo bẹ .")]},
    "帝": {"viet": "ĐẾ", "meaning_vi": "Vua.", "meo": "", "reading": "みかど", "vocab": [("帝", "みかど", "thiên hoàng"), ("上帝", "じょうてい", "thượng đế .")]},
    "締": {"viet": "ĐẾ", "meaning_vi": "Ràng buộc. Như đế giao [締交] kết bạn, đế nhân [締姻] kết dâu gia.", "meo": "", "reading": "し.まる し.まり し.める -し.め -じ.め", "vocab": [("締付", "しめづけ", "sức ép"), ("元締", "もとじめ", "người kiểm tra")]},
    "諦": {"viet": "ĐẾ, ĐỀ", "meaning_vi": "Xét kỹ, rõ. Như đế thị [諦視] coi kỹ càng.", "meo": "", "reading": "あきら.める つまびらか まこと", "vocab": [("諦め", "あきらめ", "sự từ chức; đơn xin từ chức"), ("諦める", "あきらめる", "từ bỏ; bỏ cuộc")]},
    "蹄": {"viet": "ĐỀ", "meaning_vi": "Móng chân giống thú. Như mã đề [馬蹄] vó ngựa.", "meo": "", "reading": "ひづめ", "vocab": [("蹄", "ひづめ", "móng vuốt"), ("蹄叉", "ていさ", "Xương ức")]},
    "勃": {"viet": "BỘT", "meaning_vi": "bột khởi, bột phát, đột ngột", "meo": "Cây (木) mọc từ sức mạnh (力) của mặt trời (日) vươn lên một cách bột phát.", "reading": "ボツ", "vocab": [("勃発", "ぼっぱつ", "bột phát, bùng nổ"), ("勃興", "ぼっこう", "bột hưng, trỗi dậy, nổi lên nhanh chóng")]},
    "脅": {"viet": "HIẾP", "meaning_vi": "đe dọa, uy hiếp", "meo": "HAI BÊN SƯỜN (肋) dùng SỨC MẠNH (力) để HIẾP đáp.", "reading": "おびやかす", "vocab": [("脅迫", "きょうはく", "sự cưỡng bức, sự ép buộc, sự uy hiếp"), ("脅威", "きょうい", "mối đe dọa, sự uy hiếp")]},
    "脇": {"viet": "HIẾP", "meaning_vi": "cách khác; chỗ khác; bên cạnh; hỗ trợ.", "meo": "", "reading": "わき わけ", "vocab": [("脇", "わき", "hông"), ("小脇", "こわき", "nách .")]},
    "痘": {"viet": "ĐẬU", "meaning_vi": "Lên đậu, lên mùa. Cũng gọi là thiên hoa [天花].", "meo": "", "reading": "トウ", "vocab": [("水痘", "すいとう", "bệnh thủy đậu ."), ("牛痘", "ぎゅうとう", "bệnh đậu mùa")]},
    "厨": {"viet": "TRÙ", "meaning_vi": "Cũng như chữ trù [廚].", "meo": "", "reading": "くりや", "vocab": [("厨", "くりや", "Nhà bếp"), ("庖厨", "ほうちゅう", "phòng bếp")]},
    "闘": {"viet": "ĐẤU", "meaning_vi": "Tục dùng như chữ đấu [鬬].", "meo": "", "reading": "たたか.う あらそ.う", "vocab": [("闘い", "たたかい", "trận đánh; cuộc chiến đấu"), ("闘う", "たたかう", "chiến đấu")]},
    "澄": {"viet": "TRỪNG", "meaning_vi": "Lắng trong. Nguyễn Trãi [阮 薦] : Nhất bàn lam bích trừng minh kính [一盤藍碧澄明鏡] (Vân Đồn [雲 屯]) Mặt nước như bàn xanh biếc, lắng tấm gương trong.", "meo": "", "reading": "す.む す.ます -す.ます", "vocab": [("澄む", "すむ", "trở nên trong sạch; trở nên sáng; trở nên trong"), ("澄ます", "すます", "làm sạch; làm trong sạch; lọc")]},
    "橙": {"viet": "CHANH, ĐẮNG", "meaning_vi": "Cây chanh.", "meo": "", "reading": "だいだい", "vocab": [("橙色", "だいだいいろ", "màu cam"), ("橙皮油", "とうひゆ", "dầu vỏ cam .")]},
    "噎": {"viet": "Ế", "meaning_vi": "Nghẹn.", "meo": "", "reading": "む.せる むせ.ぶ", "vocab": [("噎せる", "むせる", "chứng ho; sự ho; tiếng ho"), ("噎せ返る", "むせかえる", "lõi rau atisô")]},
    "嬉": {"viet": "HI", "meaning_vi": "Đùa bỡn, chơi.", "meo": "", "reading": "うれ.しい たの.しむ", "vocab": [("嬉しい", "うれしい", "êm lòng"), ("嬉戯", "きぎ", "sự sợ")]},
    "樹": {"viet": "THỤ", "meaning_vi": "Cây.", "meo": "", "reading": "き", "vocab": [("大樹", "だいき", "(từ Mỹ"), ("樹幹", "じゅかん", "thân (cây")]},
    "鼓": {"viet": "CỔ", "meaning_vi": "Cái trống. Đặng Trần Côn [鄧陳琨] : Cổ bề thanh động Trường Thành nguyệt [鼓鼙聲動長城月] (Chinh Phụ ngâm [征婦吟]) Tiếng trống lệnh làm rung động bóng trăng Trường Thành. Đoàn Thị Điểm dịch thơ : Trống Trường Thành lung lay bóng nguyệt.", "meo": "", "reading": "つづみ", "vocab": [("鼓", "つづみ", "trống cơm ."), ("鼓動", "こどう", "sự đập (tim); đập")]},
    "膨": {"viet": "BÀNH", "meaning_vi": "Bành hanh [膨脝] trương phềnh. Vì thế nên sự gì ngày một mở rộng hơn lên gọi là bành trướng [膨漲]. Cũng viết là [膨脹].", "meo": "", "reading": "ふく.らむ ふく.れる", "vocab": [("膨大", "ぼうだい", "lớn lên; to ra; khổng lồ"), ("膨張", "ぼうちょう", "sự bành trướng; sự tăng gia; sự mở rộng; sự giãn nở")]},
    "呂": {"viet": "LỮ, LÃ", "meaning_vi": "Luật lữ [律呂] tiếng điệu hát, xem chữ luật [律].", "meo": "", "reading": "せぼね", "vocab": [("呂律", "ろれつ", "khớp"), ("風呂", "ふろ", "bể tắm")]},
    "侶": {"viet": "LỮ", "meaning_vi": "Bạn đồng hành, bạn bè", "meo": "Nhân đứng trước một cái miệng (口) để nói chuyện với nhau, đó là bạn đồng hành (侶).", "reading": "りょ", "vocab": [("同侶", "どうりょ", "Đồng nghiệp, bạn bè"), ("僧侶", "そうりょ", "Tăng lữ, nhà sư")]},
    "宮": {"viet": "CUNG", "meaning_vi": "Cung, nhà xây tường cao mà trên uốn cong xuống gọi là cung. Nhà của vua ở và nhà để thờ thần đều gọi là cung.", "meo": "", "reading": "みや", "vocab": [("お宮", "おみや", "miếu thờ thần của Nhật Bản"), ("中宮", "ちゅうぐう", "hoàng hậu")]},
    "丑": {"viet": "SỬU, XÚ", "meaning_vi": "Một chi trong 12 chi. Từ 1 giờ đêm đến 3 giờ sáng là giờ Sửu [丑].", "meo": "", "reading": "うし", "vocab": [("癸丑", "きちゅう", "Quý Sửu ."), ("丑三つ時", "うしみつじ", "nửa đêm")]},
    "紐": {"viet": "NỮU", "meaning_vi": "Cái quạt, cái núm.", "meo": "", "reading": "ひも", "vocab": [("紐", "ひも", "dây"), ("下紐", "しもひも", "dây lưng")]},
    "慈": {"viet": "TỪ", "meaning_vi": "từ bi, nhân từ", "meo": "Có MÁI TÓC (亠) của NGƯỜI (𠆢), nói những LỜI (言, 言) từ TÂM (心) là người có lòng TỪ BI 慈.", "reading": "じ", "vocab": [("慈愛", "じあい", "tình thương, lòng từ ái"), ("慈悲", "じひ", "từ bi, lòng trắc ẩn")]},
    "磁": {"viet": "TỪ", "meaning_vi": "Từ thạch [磁石] đá nam châm. Tục dùng để gọi đồ sứ. Như từ khí [磁器] đồ sứ.", "meo": "", "reading": "ジ", "vocab": [("磁力", "じりょく", "từ học"), ("励磁", "れいじ", "sự kích thích")]},
    "滋": {"viet": "TƯ", "meaning_vi": "Thêm, càng.", "meo": "", "reading": "ジ", "vocab": [("滋味", "じみ", "đồ ăn bổ"), ("滋養", "じよう", "dinh dưỡng .")]},
    "幽": {"viet": "U", "meaning_vi": "Ần núp, sâu xa. Phàm cái gì giấu một chỗ kín không cho ai biết gọi là u. Như u cư [幽居] ở núp, u tù [幽囚] giam chỗ kín, v.v. Oán giận ai mà không nói ra được gọi là u oán [幽怨], u hận [幽恨]. Chỗ ở lặng lẽ mát mẻ gọi là u nhã [幽雅].", "meo": "", "reading": "ふか.い かす.か くら.い しろ.い", "vocab": [("幽か", "かすか", "uể oải; lả"), ("幽冥", "ゆうめい", "âm ty")]},
    "幾": {"viet": "KI, KỈ, KÍ", "meaning_vi": "Nhỏ, sự gì mới điềm ra có một tí gọi là ki. Như tri ki [知幾] biết cơ từ lúc mới có.", "meo": "", "reading": "いく- いく.つ いく.ら", "vocab": [("幾", "いく", "bao nhiêu"), ("幾つ", "いくつ", "bao nhiêu; bao nhiêu tuổi")]},
    "磯": {"viet": "KI", "meaning_vi": "Đống cát đá nổi trong nước.", "meo": "", "reading": "いそ", "vocab": [("磯", "いそ", "sỏi cát"), ("磯波", "いそなみ", "sóng nhào")]},
    "畿": {"viet": "KÌ", "meaning_vi": "Kinh kỳ [京畿] chốn kinh kì, chỗ vua thiên tử đóng.", "meo": "", "reading": "みやこ", "vocab": []},
    "緩": {"viet": "HOÃN", "meaning_vi": "Thong thả. Như hoãn bộ [緩步] bước thong thả.", "meo": "", "reading": "ゆる.い ゆる.やか ゆる.む ゆる.める", "vocab": [("緩々", "ゆる々", "chậm"), ("緩い", "ゆるい", "lỏng lẻo; chậm rãi; nhẹ nhõm; loãng; lõng bõng")]},
    "几": {"viet": "KỈ", "meaning_vi": "cái bàn", "meo": "Nhìn giống cái bàn học có 2 chân.", "reading": "つくえ", "vocab": [("几案", "きあん", "bàn giấy"), ("文机", "ふみづくえ", "bàn viết (thấp, kiểu Nhật)")]},
    "飢": {"viet": "CƠ", "meaning_vi": "Đói. Như cơ bão [飢飽] đói no. Có khi dùng như chữ ki [饑].", "meo": "", "reading": "う.える", "vocab": [("飢え", "うえ", "sự đói"), ("飢える", "うえる", "đói; thèm; khao khát; khát")]},
    "拠": {"viet": "CỨ", "meaning_vi": "Căn cứ, chiếm cứ", "meo": "", "reading": "よ.る", "vocab": [("拠る", "よる", "bởi vì; do; theo như; căn cứ vào"), ("依拠", "いきょ", "sự phụ thuộc")]},
    "抗": {"viet": "KHÁNG", "meaning_vi": "Vác.", "meo": "", "reading": "あらが.う", "vocab": [("抗争", "こうそう", "cuộc kháng chiến; sự kháng chiến; kháng chiến; phản kháng; chiến tranh; giao chiến ."), ("抗体", "こうたい", "kháng thể .")]},
    "坑": {"viet": "KHANH", "meaning_vi": "Hố. Nguyễn Trãi [阮廌] : Hân thương sinh ư ngược diễm, hãm xích tử ư họa khanh [焮蒼生於虐焰, 陷赤子於禍坑] Nướng dân đen trên ngọn lửa hung tàn, vùi con đỏ xuống dưới hầm tai vạ (Bình Ngô đại cáo [平呉大誥]).", "meo": "", "reading": "コウ", "vocab": [("坑儒", "こうじゅ", "việc chôn sống những người theo Nho học (thời Tần thủy Hoàng) ."), ("坑内", "こうない", "bằng lời nói")]},
    "杭": {"viet": "HÀNG", "meaning_vi": "Cái xuồng, cùng một nghĩa với chữ hàng [航].", "meo": "", "reading": "くい", "vocab": [("杭", "くい", "giàn ."), ("乱杭", "らんぐい", "hàng rào cọ")]},
    "燕": {"viet": "YẾN, YÊN", "meaning_vi": "Chim yến.", "meo": "", "reading": "つばめ つばくら つばくろ", "vocab": [("燕", "つばめ", "én; chim én"), ("燕巣", "つばめす", "(từ Mỹ")]},
    "凡": {"viet": "PHÀM", "meaning_vi": "Gồm, nhời nói nói tóm hết thẩy.", "meo": "", "reading": "およ.そ おうよ.そ すべ.て", "vocab": [("凡", "ぼん", "tính chất xoàng; tính chất thường; sự tầm thường; sự xoàng xĩnh"), ("凡そ", "およそ", "đại khái; ước độ; nhìn chung là; chung chung; khoảng")]},
    "帆": {"viet": "PHÀM, PHÂM", "meaning_vi": "Buồm, một thứ căng bằng vải hay bằng chiếu dựng lên trên thuyền cho gió thổi thuyền đi.", "meo": "", "reading": "ほ", "vocab": [("帆", "ほ", "thuyền buồm ."), ("出帆", "しゅっぱん", "sự đi thuyền; sự khởi hành bằng thuyền; đi thuyền; khởi hành bằng thuyền .")]},
    "汎": {"viet": "PHIẾM", "meaning_vi": "Phù phiếm.", "meo": "", "reading": "ただよ.う ひろ.い", "vocab": [("広汎", "こうはん", "rộng"), ("汎愛", "はんあい", "lòng bác ái; lòng nhân từ .")]},
    "筑": {"viet": "TRÚC", "meaning_vi": "Một thứ âm nhạc. Như cái đàn của xẩm.", "meo": "", "reading": "チク", "vocab": []},
    "伏": {"viet": "PHỤC", "meaning_vi": "Nép, nằm phục xuống.", "meo": "", "reading": "ふ.せる ふ.す", "vocab": [("伏", "ふく", "stoup"), ("伏す", "ふす", "cúi xuống lạy; bái lạy")]},
    "獄": {"viet": "NGỤC", "meaning_vi": "Ngục tù. Như hạ ngục [下獄] bắt bỏ vào nhà giam, địa ngục [地獄] theo nghĩa đen là tù ngục trong lòng đất, nơi đó tội nhân phải chịu mọi loại tra tấn do kết quả của mọi việc ác đã làm trong tiền kiếp.", "meo": "", "reading": "ゴク", "vocab": [("入獄", "にゅうごく", "Sự bị tống vào tù ."), ("典獄", "てんごく", "người cai ngục .")]},
    "獣": {"viet": "THÚ", "meaning_vi": "Thú vật", "meo": "", "reading": "けもの けだもの", "vocab": [("獣", "けだもの", "dã thú"), ("獣", "けもの", "loài thú; thú")]},
    "覇": {"viet": "BÁ", "meaning_vi": "Xưng bá.", "meo": "", "reading": "はたがしら", "vocab": [("制覇", "せいは", "sự thống trị; sự chi phối; thống trị; chi phối ."), ("覇業", "はぎょう", "sự thống trị")]},
    "撚": {"viet": "NIÊN, NIỄN, NHIÊN", "meaning_vi": "Cầm, xoe, gảy, xéo, dẫm. Cũng đọc là chữ niễn. $ Ta quen đọc là chữ nhiên.", "meo": "", "reading": "よ.る よ.れる より ひね.る", "vocab": [("撚る", "よる", "đánh chéo .")]},
    "吻": {"viet": "VẪN", "meaning_vi": "Mép.", "meo": "", "reading": "くちわき くちさき", "vocab": [("口吻", "こうふん", "sự báo cho biết; sự cho biết; điều báo cho biết"), ("吻合", "ふんごう", "sự trùng khớp")]},
    "惣": {"viet": "VẬT", "meaning_vi": "Tất cả; ý chí dân làng", "meo": "", "reading": "いそが.しい そうじて", "vocab": [("惣領", "そうりょう", "sự cai trị")]},
    "忽": {"viet": "HỐT", "meaning_vi": "bỗng nhiên, đột ngột", "meo": "Tim (心) bị cắt (勿) làm đôi vì quá bất ngờ, hốt hoảng.", "reading": "こつ", "vocab": [("忽如", "こつじょ", "bỗng nhiên, đột ngột"), ("忽地", "たちまち", "ngay lập tức, trong khoảnh khắc")]},
    "惚": {"viet": "HỐT", "meaning_vi": "Hoảng hốt [恍惚] thấy không đích xác.", "meo": "", "reading": "ほけ.る ぼ.ける ほ.れる", "vocab": [("惚け", "ぼけ", "sự lão suy; sự lẩn thẩn ."), ("恍惚", "こうこつ", "trạng thái mê ly")]},
    "葱": {"viet": "THÔNG", "meaning_vi": "Hành. Chỗ tóp trắng gọi là thông bạch [葱白].", "meo": "", "reading": "ねぎ", "vocab": [("葱", "ねぎ", "hành ."), ("浅葱", "あさつき", "Cây hẹ tây .")]},
    "賜": {"viet": "TỨ", "meaning_vi": "Cho, trên cho dưới gọi là tứ. Như hạ tứ [下賜] ban cho kẻ dưới, sủng tứ [寵賜] vua yêu mà ban cho, v.v.", "meo": "", "reading": "たまわ.る たま.う たも.う", "vocab": [("賜う", "たまう", "sự cho"), ("賜る", "たまわる", "ban thưởng")]},
    "錫": {"viet": "TÍCH", "meaning_vi": "Thiếc (Stannum, St); sắc trắng như bạc, chất mềm chóng chảy, vì thế nên người ta hay dùng để tráng mặt đồ đồng đồ sắt cho đẹp.", "meo": "", "reading": "すず たま.う", "vocab": [("錫", "すず", "thiếc ."), ("錫杖", "しゃくじょう", "thiếc lá .")]},
    "罵": {"viet": "MẠ", "meaning_vi": "Mắng chửi. Nguyễn Du [阮攸] : Tặc cốt thiên niên mạ bất tri [賊骨天年罵不知] (Thất thập nhị nghi trủng [七十二疑冢]) Nắm xương giặc (chỉ Tào Tháo [曹操]) nghìn năm bị chửi rủa mà không biết.", "meo": "", "reading": "ののし.る", "vocab": [("罵る", "ののしる", "chửi"), ("罵倒", "ばとう", "sự lạm dụng")]},
    "篤": {"viet": "ĐỐC", "meaning_vi": "Hậu, thuần nhất không có cái gì xen vào gọi là đốc. Như đốc tín [篤信] dốc một lòng tin, đôn đốc [敦篤] dốc một lòng chăm chỉ trung hậu, v.v. Luận ngữ [論語] : Đốc tín hiếu học, thủ tử thiện đạo [篤信好學, 守死善道] (Thái Bá [泰伯]) Vững tin ham học, giữ đạo tới chết.  Ốm nặng, bệnh tình trầm trọng.", "meo": "", "reading": "あつ.い", "vocab": [("篤い", "あつい", "đứng đắn"), ("篤信", "とくしん", "Sự tận tâm .")]},
    "架": {"viet": "GIÁ", "meaning_vi": "Cái giá. Như y giá [衣架] cái giá mắc áo, thư giá [書架] cái giá sách, v.v.", "meo": "", "reading": "か.ける か.かる", "vocab": [("架", "か", "những đám mây trôi giạt"), ("刀架", "とうか", "giá treo gươm; giá để đao .")]},
    "賀": {"viet": "HẠ", "meaning_vi": "Đưa đồ mừng. Như hạ lễ [賀禮] đồ lễ mừng.", "meo": "", "reading": "ガ", "vocab": [("賀宴", "がえん", "tiệc lớn"), ("年賀", "ねんが", "sự mừng năm mới; lễ tết")]},
    "迦": {"viet": "GIÀ, CA", "meaning_vi": "Thích Già [釋迦] đức Thích Già là vị tổ sáng lập ra Phật giáo. Cũng đọc là chữ ca.", "meo": "", "reading": "カ ケ", "vocab": [("釈迦", "しゃか", "thích ca"), ("お釈迦", "おしゃか", "phá vỡ hợp đồng; hủy bỏ điều khoản")]},
    "駕": {"viet": "GIÁ", "meaning_vi": "Đóng xe ngựa (đóng ngựa vào xe).", "meo": "", "reading": "かご が.する しのぐ のる", "vocab": [("凌駕", "りょうが", "vượt hơn"), ("出駕", "でが", "sự mọc mộng")]},
    "焚": {"viet": "PHẦN, PHẪN", "meaning_vi": "Đốt. Như phần hương [焚香] đốt hương.", "meo": "", "reading": "た.く や.く やきがり", "vocab": [("焚く", "たく", "thiêu đốt; đốt (lửa)"), ("焚火", "たきび", "lửa mừng")]},
    "淋": {"viet": "LÂM", "meaning_vi": "Ngâm nước.", "meo": "", "reading": "さび.しい さみ.しい", "vocab": [("淋巴", "りんぱ", "bạch huyết"), ("淋しい", "さびしい", "vắng vẻ")]},
    "麻": {"viet": "MA", "meaning_vi": "Đại ma [大麻] cây gai. Có khi gọi là hỏa ma [火麻] hay hoàng ma [黃麻]. Có hai giống đực và cái, giống đực gọi là mẫu ma [牡麻], giống cái gọi là tử ma [子麻]. Sang tiết xuân phân mới gieo hạt, trước sau tiết hạ chí mới nở hoa, sắc trắng xanh xanh. Gai đực có năm nhị, gai cái có một nhị. Gai đực thì khi hoa rụng hết liền nhổ, ngâm nước bóc lấy vỏ, mềm nhũn mà có thớ dài, dùng để dệt vải thưa. Gai cái thì đến mùa thu mới cắt, bóc lấy hạt rồi mới đem ngâm, dùng để dệt sô gai, vì nó đen và xù xì nên chỉ dùng làm đồ tang và túi đựng đồ thôi. Hạt nó ăn được.", "meo": "", "reading": "あさ", "vocab": [("麻", "あさ", "vải lanh; cây lanh; cây gai"), ("乱麻", "らんま", "tình trạng vô chính phủ")]},
    "摩": {"viet": "MA", "meaning_vi": "Xoa xát. Như ma quyền sát chưởng [摩拳擦掌] xoa nắm tay xát bàn tay.", "meo": "", "reading": "ま.する さす.る す.る", "vocab": [("摩る", "さする", "xoa bóp; nặn"), ("削摩", "けずま", "sự lột trần (quần áo")]},
    "磨": {"viet": "MA, MÁ", "meaning_vi": "Mài, xát. Nghiên cứu học vấn gọi là thiết tha trác ma [切磋琢磨]. Tuân Tử [荀子] : Nhân chi ư văn học dã, do ngọc chi ư trác ma dã [人之於文學也, 猶玉之於琢磨也] Người học văn, cũng như ngọc phải giũa phải mài vậy.", "meo": "", "reading": "みが.く す.る", "vocab": [("磨き", "みがき", "Polish  Ba lan"), ("磨く", "みがく", "đánh bóng; làm sáng bóng; mài bóng; mài")]},
    "魔": {"viet": "MA", "meaning_vi": "Ma quỷ. Các cái làm cho người ta mê muội, làm mất lòng đạo đều gọi là ma cả. Như yêu ma [妖魔] ma quái.", "meo": "", "reading": "マ", "vocab": [("魔", "ま", "ma quỷ"), ("魔力", "まりょく", "ma lực .")]},
    "暦": {"viet": "LỊCH", "meaning_vi": "Lịch, tấm lịch; niên đại", "meo": "", "reading": "こよみ りゃく", "vocab": [("暦", "こよみ", "niên lịch; niên giám; lịch ."), ("暦年", "れきねん", "thời gian")]},
    "襟": {"viet": "KHÂM", "meaning_vi": "Vạt áo, cổ áo. Nguyễn Trãi [阮廌] : Thi thành ngã diệc lệ triêm khâm [詩成我亦淚沾襟] (Đề Hà Hiệu Úy \"Bạch vân tư thân\" [題何校尉白雲思親]) Thơ làm xong, nước mắt ta cũng ướt đẫm vạt áo.", "meo": "", "reading": "えり", "vocab": [("襟", "えり", "cổ áo"), ("襟元", "えりもと", "cổ (phần trước cổ)")]},
    "讃": {"viet": "", "meaning_vi": "praise, title on a picture", "meo": "", "reading": "ほ.める たた.える", "vocab": [("讃える", "たたえる", "tán dương"), ("讃歌", "さんか", "bài tán dương")]},
    "潜": {"viet": "TIỀM", "meaning_vi": "Giản thể của chữ [潛].", "meo": "", "reading": "ひそ.む もぐ.る かく.れる くぐ.る ひそ.める", "vocab": [("潜む", "ひそむ", "ẩn núp; trốn; ẩn giấu"), ("潜り", "もぐり", "việc lặn dưới nước")]},
    "笹": {"viet": "", "meaning_vi": "bamboo grass", "meo": "", "reading": "ささ", "vocab": [("笹巻(ベトナムの食品)", "ささまき（べとなむのしょくひん）", "bánh chưng .")]},
    "貰": {"viet": "THẾ", "meaning_vi": "Vay, cho thuê đồ cũng gọi là thế. Như thế mãi [貰買] mua chịu.", "meo": "", "reading": "もら.う", "vocab": [("貰う", "もらう", "nhận ."), ("貰い手", "もらいて", "người nhận")]},
    "喋": {"viet": "ĐIỆP", "meaning_vi": "Điệp điệp [喋喋] nói lem lém.", "meo": "", "reading": "しゃべ.る ついば.む", "vocab": [("喋る", "しゃべる", "nói chuyện; tán gẫu"), ("お喋り", "おしゃべり", "hay nói; hay chuyện; lắm mồm")]},
    "諜": {"viet": "ĐIỆP", "meaning_vi": "Dò xét, người đi dò thám quân lính bên giặc gọi là điệp, tục gọi là tế tác [細作].", "meo": "", "reading": "ちょう.ずる うかが.う しめ.す", "vocab": [("諜報", "ちょうほう", "Thông tin bí mật ."), ("間諜", "かんちょう", "/'spaiə/")]},
    "蝶": {"viet": "ĐIỆP", "meaning_vi": "Con bướm. Như sứ điệp [使蝶] con bướm trao tin, chỉ thư từ trao đổi trai gái.", "meo": "", "reading": "チョウ", "vocab": [("蝶", "ちょう", "bướm ."), ("蝶々", "ちょうちょう", "bướm")]},
    "姻": {"viet": "NHÂN", "meaning_vi": "Nhà trai. Bố vợ gọi là hôn [婚], bố chồng gọi là nhân [姻]. Hai chữ này nay hay dùng lẫn lộn, nên gọi sự kết hôn là đế nhân [締姻].", "meo": "", "reading": "イン", "vocab": [("婚姻", "こんいん", "hôn nhân ."), ("姻戚", "いんせき", "mối quan hệ")]},
    "咽": {"viet": "YẾT, YẾN, Ế", "meaning_vi": "Cổ họng. Như yết hầu [咽喉] cổ họng.", "meo": "", "reading": "むせ.ぶ むせ.る のど の.む", "vocab": [("咽", "のんど", "họng"), ("咽る", "のんどる", "chứng ho; sự ho; tiếng ho")]},
    "恩": {"viet": "ÂN", "meaning_vi": "Ơn. yêu mà giúp đỡ mà ban cho cái gì gọi là ân.", "meo": "", "reading": "オン", "vocab": [("恩", "おん", "ân; ân nghĩa; ơn; ơn nghĩa"), ("主恩", "しゅおん", "âm chủ")]},
    "冒": {"viet": "MẠO, MẶC", "meaning_vi": "Trùm đậy.", "meo": "", "reading": "おか.す", "vocab": [("冒す", "おかす", "đương đầu với; liều; mạo phạm; đe dọa"), ("冒とく", "ぼうとく", "lời báng bổ; sự nguyền rủa")]},
    "至": {"viet": "CHÍ", "meaning_vi": "Đến. Như tân chí như quy [賓至如歸] khách đến như về chợ.", "meo": "", "reading": "いた.る", "vocab": [("至り", "いたり", "đầu"), ("至る", "いたる", "đạt tới; đạt đến")]},
    "緻": {"viet": "TRÍ", "meaning_vi": "Tỉ mỉ, kín, kĩ. Như công trí [工緻] khéo mà kĩ, tinh trí [精緻] tốt bền, trí mật [緻密] đông đặc, liền sít.", "meo": "", "reading": "こまか.い", "vocab": [("緻密", "ちみつ", "phút"), ("巧緻", "こうち", "phức tạp")]},
    "姪": {"viet": "ĐIỆT", "meaning_vi": "Cháu, tiếng xưng hô đối với chú bác.", "meo": "", "reading": "めい おい", "vocab": [("姪", "めい", "cháu gái .")]},
    "蛭": {"viet": "ĐIỆT", "meaning_vi": "Con đỉa. Có khi gọi là thủy điệt [水蛭]. Thứ lớn gọi là mã điệt [馬蛭], tục gọi là mã hoàng [馬蝗].", "meo": "", "reading": "ひる", "vocab": [("蛭", "ひる", "con đỉa")]},
    "窒": {"viet": "TRẤT", "meaning_vi": "nghẹt, ngạt, bế tắc", "meo": "Nhà (宀) đến rồi, chất (质) đầy nhà gây ngạt thở.", "reading": "ちっ", "vocab": [("窒素", "ちっそ", "khí nitơ"), ("窒息", "ちっそく", "sự ngạt thở")]},
    "尋": {"viet": "TẦM", "meaning_vi": "Tìm.", "meo": "", "reading": "たず.ねる ひろ", "vocab": [("千尋", "ちひろ", "không có đáy"), ("尋問", "じんもん", "câu hỏi; sự tra hỏi; tra hỏi .")]},
    "浸": {"viet": "TẨM", "meaning_vi": "Tẩm, ngâm.", "meo": "", "reading": "ひた.す ひた.る", "vocab": [("浸す", "ひたす", "đắm đuối"), ("浸る", "ひたる", "bị thấm ướt; bị ngập nước; ngập chìm")]},
    "擬": {"viet": "NGHĨ", "meaning_vi": "Mô phỏng, bắt chước", "meo": "Nghi ngờ (nghi) + mũi tên (矢) + tay (Thủ): Nghi ngờ người cầm mũi tên bằng tay để bắt chước.", "reading": "ぎ", "vocab": [("擬音", "ぎおん", "Âm thanh mô phỏng"), ("模擬", "もぎ", "Mô phỏng")]},
    "凝": {"viet": "NGƯNG", "meaning_vi": "Đọng lại. Chất lỏng đọng lại gọi là ngưng.", "meo": "", "reading": "こ.る こ.らす こご.らす こご.らせる こご.る", "vocab": [("凝", "こご", "sự đông vì lạnh"), ("凝り", "こり", "sự phồng ra")]},
    "轄": {"viet": "HẠT", "meaning_vi": "Cái đinh chốt xe, cái chốt cắm ngoài đầu trục cho bánh xe không trụt ra được.", "meo": "", "reading": "くさび", "vocab": [("所轄", "しょかつ", "phạm vi quyền lực; quyền hạn xét xử; quyền thực thi pháp lý"), ("直轄", "ちょっかつ", "sự trực thuộc")]},
    "繕": {"viet": "THIỆN", "meaning_vi": "Sửa, chữa. Như tu thiện [修繕] sửa sang. Sửa sang đồ binh gọi là chinh thiện [征繕] hay chỉnh thiện [整繕].", "meo": "", "reading": "つくろ.う", "vocab": [("繕い", "つくろい", "vật được vá/tu sữa/sửa chữa"), ("繕う", "つくろう", "sắp xếp gọn gàng sạch sẽ; sắp xếp đúng vị trí")]},
    "膳": {"viet": "THIỆN", "meaning_vi": "Cỗ ăn.", "meo": "", "reading": "かしわ すす.める そな.える", "vocab": [("膳", "ぜん", "khay"), ("お膳", "おぜん", "khay bốn chân; mâm bốn chân")]},
    "衡": {"viet": "HÀNH", "meaning_vi": "cân, thăng bằng, đo lường", "meo": "Chim (隹) đi trong thành (行) mà cứ phải đo đi đo lại để giữ thăng bằng ngang bằng (衡)", "reading": "こう", "vocab": [("均衡", "きんこう", "cân bằng, quân bình"), ("度量衡", "どりょうこう", "đo lường, cân đo")]},
    "桁": {"viet": "HÀNH, HÀNG, HÃNG", "meaning_vi": "Ốc hành [屋桁] cái rầm gỗ.", "meo": "", "reading": "けた", "vocab": [("桁", "けた", "nhịp cầu; bi bàn tính; ký tự; chữ số"), ("二桁", "ふたけた", "Số hai chữ số .")]},
    "酬": {"viet": "THÙ", "meaning_vi": "Mời rượu. Chủ mời khách uống gọi là thù [酬], khách rót lại chủ gọi là tạc [酢]. Vì thế nên ở đời phải đi lại với nhau để tỏ tình thân đều gọi là thù tạc [酬酢].", "meo": "", "reading": "むく.いる", "vocab": [("酬い", "むくい", "sự trở lại"), ("報酬", "ほうしゅう", "sự báo thù")]},
    "馴": {"viet": "THUẦN", "meaning_vi": "quen, thuần hóa", "meo": "Con NGỰA (馬) đi theo SÔNG (川) riết rồi cũng QUEN.", "reading": "な", "vocab": [("馴れる", "なれる", "quen, quen thuộc"), ("馴らす", "ならす", "thuần hóa, làm cho quen")]},
    "硫": {"viet": "LƯU", "meaning_vi": "Lưu hoàng [硫黃] lưu hoàng, ta thường gọi là diêm vàng.", "meo": "", "reading": "リュウ", "vocab": [("硫化", "りゅうか", "sự cho ngấm lưu huỳnh; sự xông lưu huỳnh"), ("硫酸", "りゅうさん", "axit sunphuric")]},
    "荒": {"viet": "HOANG", "meaning_vi": "Bỏ hoang, đất đầy những cỏ gọi là hoang. Nên ruộng chưa vỡ cỏ, chưa cầy cấy được gọi là hoang điền [荒田] ruộng hoang. Khai hoang [開荒], khẩn hoang [墾荒] đều nghĩa là khai khẩn ruộng bỏ hoang cả. Ruộng vẫn cấy được, mà vì tai biến lúa không chín được, cũng gọi là hoang. Như thủy hoang [水荒] bị lụt, hạn hoang [旱荒] đại hạn.", "meo": "", "reading": "あら.い あら- あ.れる あ.らす -あ.らし すさ.む", "vocab": [("荒い", "あらい", "gấp gáp; dữ dội; khốc liệt; thô bạo"), ("荒す", "あらす", "phá huỷ; gây thiệt hại; phá")]},
    "慌": {"viet": "HOẢNG", "meaning_vi": "Lờ mờ. Như hoảng hốt [慌惚].", "meo": "", "reading": "あわ.てる あわ.ただしい", "vocab": [("恐慌", "きょうこう", "khủng hoảng; sự kinh hoàng; sự thất kinh; sự khiếp đảm; sự rụng rời;"), ("慌てる", "あわてる", "trở nên lộn xộn; vội vàng; luống cuống; bối rối")]},
    "暁": {"viet": "HIỂU", "meaning_vi": "Bình minh", "meo": "", "reading": "あかつき さと.る", "vocab": [("暁", "あかつき", "bình minh"), ("今暁", "こんぎょう", "sáng nay .")]},
    "疎": {"viet": "SƠ", "meaning_vi": "Tục dùng như chữ sơ [疏].", "meo": "", "reading": "うと.い うと.む まば.ら", "vocab": [("疎い", "うとい", "vô tư; không vụ lợi"), ("疎か", "おろそか", "thờ ơ; không quan tâm; lãng quên; lãng phí; sao lãng")]},
    "辣": {"viet": "LẠT", "meaning_vi": "Cay. Như toan điềm khổ lạt [酸甜苦辣] chua ngọt đắng cay.", "meo": "", "reading": "から.い", "vocab": [("悪辣", "あくらつ", "gian ác ."), ("辣腕", "らつわん", "sự khôn; tính khôn ngoan; tính sắc sảo")]},
    "嗽": {"viet": "THẤU", "meaning_vi": "Ho nhổ (ho có đờm).", "meo": "", "reading": "すす.ぐ ゆす.ぐ くちすす.ぐ うがい", "vocab": [("嗽", "うがい", "thuốc súc miệng"), ("含嗽", "うがい", "sự súc miệng")]},
    "漱": {"viet": "SẤU, THẤU", "meaning_vi": "Súc miệng.", "meo": "", "reading": "くちすす.ぐ くちそそ.ぐ うがい すす.ぐ", "vocab": [("漱ぐ", "すすぐ", "súc ."), ("口を漱ぐ", "くちをすすぐ", "súc miệng .")]},
    "勅": {"viet": "SẮC", "meaning_vi": "Tục dùng như chữ sắc [敕] nghĩa là răn bảo.", "meo": "", "reading": "いまし.める みことのり", "vocab": [("勅", "ちょく", "tờ sắc (của nhà vua"), ("勅令", "ちょくれい", "Sắc lệnh (hoàng đế)")]},
    "爪": {"viet": "TRẢO", "meaning_vi": "Móng chân, móng tay.", "meo": "", "reading": "つめ つま-", "vocab": [("爪", "つめ", "móng"), ("爪先", "つまさき", "đầu ngón chân .")]},
    "瓜": {"viet": "QUA", "meaning_vi": "Dưa, các thứ dưa có quả đều gọi là qua.", "meo": "", "reading": "うり", "vocab": [("瓜", "うり", "dưa; bầu; bí"), ("南瓜", "かぼちゃ", "bí ngô; quả bí ngô; bí rợ; bí đỏ")]},
    "狐": {"viet": "HỒ", "meaning_vi": "Con hồ (con cáo). Da nó lột may áo ấm gọi là hồ cừu [狐裘].", "meo": "", "reading": "きつね", "vocab": [("狐", "きつね", "cáo; chồn"), ("古狐", "ふるぎつね", "dân kỳ cựu (ở nơi nào")]},
    "孤": {"viet": "CÔ", "meaning_vi": "Mồ côi. Mồ côi cha sớm gọi là tảo cô [早孤].", "meo": "", "reading": "コ", "vocab": [("孤", "こ", "mồ côi"), ("孤児", "こじ", "cô nhi; trẻ mồ côi")]},
    "弧": {"viet": "HỒ, O", "meaning_vi": "Cái cung gỗ. Như tang hồ [桑弧] cung dâu. Lễ ngày xưa đẻ con trai thì treo cái cung gỗ ở bên cửa tay trái, tỏ ý con trai phải có chí bốn phương, vì thế nên đẻ con trai gọi là huyền hồ [懸弧] treo cung.", "meo": "", "reading": "コ", "vocab": [("弧", "こ", "hình cung"), ("円弧", "えんこ", "cung tròn")]},
    "派": {"viet": "PHÁI", "meaning_vi": "Dòng nước. Nguyễn Du [阮攸] : Thiên Hoàng cự phái cửu thiên lí [天潢巨派九千里] (Hoàng Hà [黄河]) Nhánh lớn của Sông Trời, dài chín ngàn dặm.", "meo": "", "reading": "ハ", "vocab": [("派", "は", "nhóm; bè phái; bè cánh"), ("一派", "いっぱ", "đàn cá")]},
    "赦": {"viet": "XÁ", "meaning_vi": "Tha, tha cho kẻ có tội gọi là xá. Như đại xá thiên hạ [大赦天下] cả tha cho thiên hạ. Mỗi khi vua lên ngôi hay có việc mừng lớn của nhà vua thì tha tội cho các tù phạm và thuế má gọi là đại xá thiên hạ.", "meo": "", "reading": "シャ", "vocab": [("赦す", "ゆるす", "xá ."), ("赦免", "しゃめん", "sự tha thứ")]},
    "嚇": {"viet": "HÁCH", "meaning_vi": "đe dọa, hăm dọa", "meo": "Miệng (口) to như con hổ (虎) để hách dịch, hăm dọa.", "reading": "おど.す", "vocab": [("威嚇", "いかく", "uy hiếp, đe dọa"), ("恫喝", "どうかつ", "hăm dọa, đe nẹt")]},
    "亦": {"viet": "DIỆC", "meaning_vi": "Cũng, tiếng giúp lời nói. Như trị diệc tiến, loạn diệc tiến [治亦進亂亦進] trị cũng tiến lên, loạn cũng tiến lên.", "meo": "", "reading": "また", "vocab": [("亦", "また", "cũng")]},
    "蛮": {"viet": "MAN", "meaning_vi": "Giản thể của chữ 蠻", "meo": "", "reading": "えびす", "vocab": [("蛮", "ばん", "người dã man; người man rợ ."), ("蛮人", "ばんじん", "người hoang dã; người man rợ .")]},
    "衷": {"viet": "TRUNG, TRÚNG", "meaning_vi": "Tốt, lành.", "meo": "", "reading": "チュウ", "vocab": [("衷心", "ちゅうしん", "sự thật tâm ."), ("衷情", "ちゅうじょう", "sự thật tâm; cảm xúc bên trong .")]},
    "哀": {"viet": "AI", "meaning_vi": "Thương.", "meo": "", "reading": "あわ.れ あわ.れむ かな.しい", "vocab": [("哀れ", "あわれ", "đáng thương; buồn thảm; bi ai"), ("哀傷", "あいしょう", "Buồn rầu; sự đau buồn")]},
    "衰": {"viet": "SUY", "meaning_vi": "suy yếu, suy tàn", "meo": "Áo (衣) trùm đầu đội mũ, trên đầu lại có hai giọt nước (亠) vì SUY sụp.", "reading": "おとろ.える", "vocab": [("衰弱", "すいじゃく", "suy nhược"), ("衰退", "すいたい", "suy thoái")]},
    "喪": {"viet": "TANG, TÁNG", "meaning_vi": "Lễ tang. Như cư tang [居喪] để tang, điếu tang [弔喪] viếng kẻ chết, v.v.", "meo": "", "reading": "も", "vocab": [("喪", "も", "quần áo tang; đồ tang"), ("喪中", "もちゅう", "đang có tang")]},
    "畏": {"viet": "ÚY", "meaning_vi": "Sợ. Sự gì chưa xảy ra mới tưởng tượng cũng đáng sợ gọi là cụ [懼], sự đã xảy đến phải nhận là đáng sợ gọi là úy [畏].", "meo": "", "reading": "おそ.れる かしこま.る かしこ かしこ.し", "vocab": [("畏怖", "いふ", "ván cánh bánh xe nước"), ("畏くも", "かしこくも", "hoà nhã")]},
    "隈": {"viet": "ÔI", "meaning_vi": "Chỗ núi, nước uống cong. Như sơn ôi [山隈] khuỷu núi.", "meo": "", "reading": "くま すみ", "vocab": [("隈なく", "くまなく", "khắp nơi; mọi nơi; tất cả các nơi"), ("界隈", "かいわい", "hàng xóm")]},
    "猥": {"viet": "ỔI", "meaning_vi": "dâm ô, tục tĩu", "meo": "Bên trái là bộ NỮ 女, bên phải là ỦY 委 (gái mà cứ khom lưng làm theo ý người khác thì dễ bị sàm sỡ).", "reading": "わい", "vocab": [("猥褻", "わいせつ", "dâm ô, tục tĩu"), ("猥雑", "わいざつ", "lộn xộn, tạp nham, không tao nhã")]},
    "嬢": {"viet": "NƯƠNG", "meaning_vi": "Cũng như chữ nương [娘].", "meo": "", "reading": "むすめ", "vocab": [("嬢", "じょう", "cô gái ."), ("令嬢", "れいじょう", "cô gái; lệnh nương .")]},
    "譲": {"viet": "NHƯỢNG", "meaning_vi": "Nhượng bộ.", "meo": "", "reading": "ゆず.る", "vocab": [("譲る", "ゆずる", "bàn giao (quyền sở hữu tài sản)"), ("譲与", "じょうよ", "sự di chuyển")]},
    "壌": {"viet": "NHƯỠNG", "meaning_vi": "Thổ nhưỡng.", "meo": "", "reading": "つち", "vocab": [("土壌", "どじょう", "đất cát"), ("天壌", "てんじょう", "Thiên đàng và mặt đất .")]},
    "醸": {"viet": "NHƯỠNG", "meaning_vi": "Gây nên", "meo": "", "reading": "かも.す", "vocab": [("醸す", "かもす", "làm lên men; ủ; chế"), ("醸成", "じょうせい", "việc lên men; sự ủ rượu")]},
    "棚": {"viet": "BẰNG", "meaning_vi": "Gác, nhà rạp.", "meo": "", "reading": "たな -だな", "vocab": [("棚", "たな", "cái giá"), ("棚卸", "たなおろし", "kiểm kê .")]},
    "尺": {"viet": "XÍCH", "meaning_vi": "Thước, mười tấc là một thước.", "meo": "", "reading": "シャク", "vocab": [("尺", "しゃく", "cái thước"), ("三尺", "さんしゃく", "cạp (dải vải tạo thành chỗ eo lưng của quần áo")]},
    "釈": {"viet": "THÍCH", "meaning_vi": "chú thích. giải thích", "meo": "", "reading": "とく す.てる ゆる.す", "vocab": [("会釈", "えしゃく", "sự cúi chào; gật đầu"), ("保釈", "ほしゃく", "bảo lãnh")]},
    "択": {"viet": "TRẠCH", "meaning_vi": "Chọn lựa.", "meo": "", "reading": "えら.ぶ", "vocab": [("択一", "たくいつ", "sự lựa chọn một trong hai (vật"), ("採択", "さいたく", "sự lựa chọn .")]},
    "尽": {"viet": "TẬN", "meaning_vi": "cạn kiệt, dốc hết sức", "meo": "Người (人) đứng hết sức (一) rồi giơ hai tay lên (丷, giống hai sừng con dê) thì là tận sức.", "reading": "つく", "vocab": [("尽きる", "つきる", "cạn kiệt, hết"), ("尽くす", "つくす", "dốc hết sức, phục vụ")]},
    "剛": {"viet": "CƯƠNG", "meaning_vi": "Cứng, bền. Cố chấp không nghe ai can gọi là cương phức [剛愎].", "meo": "", "reading": "ゴウ", "vocab": [("剛健", "ごうけん", "khoẻ mạnh; chắc chắn; vững chãi"), ("内剛", "ないごう", "nội nhu ngoại cương")]},
    "鋼": {"viet": "CƯƠNG", "meaning_vi": "Thép. Sắt luyện kỹ gọi là cương.", "meo": "", "reading": "はがね", "vocab": [("鋼", "はがね", "thép ."), ("丸鋼", "まるこう", "thép tròn .")]},
    "綱": {"viet": "CƯƠNG", "meaning_vi": "Giường lưới. Lưới có giường mới kéo được các mắt, cho nên cái gì mà có thống hệ không thể rời được đều gọi là cương. Cương thường [綱常] đạo thường của người gồm : tam cương [三綱] (quân thần, phụ tử, phu phụ [君臣, 父子, 夫婦]) và ngũ thường [五常] (nhân, lễ, nghĩa, trí, tín [仁義禮智信]). Cương kỷ [綱紀] giường mối, v.v.", "meo": "", "reading": "つな", "vocab": [("綱", "つな", "dây thừng; sợi dây thừng; dây chão ."), ("亜綱", "あつな", "phân lớp")]},
    "礫": {"viet": "LỊCH", "meaning_vi": "Đá vụn, đá sỏi. Liễu Tông Nguyên [柳宗元] : Kỳ bàng đa nham đỗng, kỳ hạ đa bạch lịch [其旁多巖洞, 其下多白礫] (Viên gia kiệt kí [袁家渴記]) ở bên có nhiều núi cao hang động, ở dưới nhiều đá nhỏ sỏi trắng.", "meo": "", "reading": "つぶて こいし", "vocab": [("礫", "つぶて", "sự ném đá ."), ("礫岩", "れきがん", "đá cuội .")]},
    "牙": {"viet": "NHA", "meaning_vi": "Răng to.", "meo": "", "reading": "きば は", "vocab": [("牙", "きば", "ngà"), ("牙城", "がじょう", "đồn")]},
    "芽": {"viet": "NHA", "meaning_vi": "Mầm. Như đậu nha [荳芽] mầm đậu.", "meo": "", "reading": "め", "vocab": [("芽", "め", "búp"), ("出芽", "しゅつが", "sự mọc mộng")]},
    "冴": {"viet": "NGÀ", "meaning_vi": "be clear, serene, cold, skilful", "meo": "", "reading": "さ.える こお.る ひ.える", "vocab": [("冴える", "さえる", "khéo léo"), ("腕の冴え", "うでのさえ", "Sự khéo tay; tài khéo léo; sự khéo léo; khéo")]},
    "雅": {"viet": "NHÃ", "meaning_vi": "Chính, một lối thơ ca dùng vào nhạc ngày xưa. Như Thi Kinh [詩經] có Đại nhã [大雅], Tiểu nhã [小雅] ý nói những khúc ấy mới là khúc hát chính đính vậy.", "meo": "", "reading": "みや.び", "vocab": [("優雅", "ゆうが", "sự dịu dàng; sự thanh lịch"), ("典雅", "てんが", "sự thanh lịch; sự thanh nhã; sự nhã nhặn")]},
    "邪": {"viet": "TÀ, DA", "meaning_vi": "Lệch, cong.", "meo": "", "reading": "よこし.ま", "vocab": [("邪", "よこしま", "xấu"), ("邪宗", "じゃしゅう", "dị giáo .")]},
    "穿": {"viet": "XUYÊN", "meaning_vi": "Thủng lỗ.", "meo": "", "reading": "うが.つ は.く", "vocab": [("穿く", "はく", "sự mang; sự dùng; sự mặc"), ("穿つ", "うがつ", "mũi khoan; máy khoan")]},
    "既": {"viet": "KÍ", "meaning_vi": "Đã, rồi. Cũng như chữ kí [旣]. Tô Thức [蘇軾] : Bất tri đông phương chi kí bạch [不知東方之既白] (Tiền Xích Bích phú [前赤壁賦 ]) Không biết phương đông đã sáng bạch.", "meo": "", "reading": "すで.に", "vocab": [("既に", "すでに", "đã; đã muộn; đã rồi ."), ("既刊", "きかん", "người đun")]},
    "慨": {"viet": "KHÁI", "meaning_vi": "Tức giận bồn chồn. Như khảng khái [慷慨].", "meo": "", "reading": "ガイ", "vocab": [("慨嘆", "がいたん", "lời than vãn; lời than thở; sự than vãn; sự than thở; than vãn; than thở"), ("感慨", "かんがい", "cảm khái; sự cảm khái; cảm giác; tâm trạng; cảm xúc")]},
    "概": {"viet": "KHÁI", "meaning_vi": "Gạt phẳng.", "meo": "", "reading": "おおむ.ね", "vocab": [("概ね", "おおむね", "hầu hết"), ("大概", "たいがい", "sự bao quát; sự nhìn chung; sự chủ yếu")]},
    "郡": {"viet": "QUẬN", "meaning_vi": "Quận. Một tên riêng để gọi khu đất đã chia giới hạn. Nước ta ngày xưa chia làm 12 quận. Như quận huyện [郡縣] quận và huyện, hai đơn vị hành chánh trong nước, cũng để chỉ chung lãnh thổ đất nước.", "meo": "", "reading": "こおり", "vocab": [("郡", "ぐん", "huyện"), ("郡県", "ぐんけん", "tỉnh và huyện .")]},
    "茸": {"viet": "NHUNG, NHŨNG", "meaning_vi": "Mầm nõn, lá nõn.", "meo": "", "reading": "きのこ たけ しげ.る", "vocab": [("茸", "きのこ", "nấm"), ("毒茸", "どくたけ", "nấm mũ độc")]},
    "餌": {"viet": "NHỊ", "meaning_vi": "Bánh bột, các chất bổ cho người ốm ăn gọi là dược nhị [葯餌].", "meo": "", "reading": "え えば えさ もち", "vocab": [("餌", "えさ", "mồi; đồ ăn cho động vật; thức ăn gia súc gia cầm"), ("好餌", "こうじ", "bate")]},
    "摂": {"viet": "NHIẾP", "meaning_vi": "Vén lên.Tô Thức [蘇軾]  : Dư nãi nhiếp y nhi thướng [予乃攝衣而上]  (Hậu Xích Bích phú [後赤壁賦]) Tôi bèn vén áo mà lên.", "meo": "", "reading": "おさ.める かね.る と.る", "vocab": [("兼摂", "けんせつ", "sự xây dựng"), ("包摂", "ほうせつ", "sự xếp")]},
    "敢": {"viet": "CẢM", "meaning_vi": "Tiến lên. Như dũng cảm [勇敢] mạnh bạo tiến lên.", "meo": "", "reading": "あ.えて あ.えない あ.えず", "vocab": [("勇敢", "ゆうかん", "can đảm"), ("敢えて", "あえて", "dám")]},
    "巧": {"viet": "XẢO", "meaning_vi": "Khéo. Nguyễn Du [阮攸] : Thiên cơ vạn xảo tận thành không [千機萬巧盡成空] (Đồng Tước đài [銅雀臺]) Rốt cuộc muôn khéo nghìn khôn cũng thành không tất cả.", "meo": "", "reading": "たく.み たく.む うま.い", "vocab": [("巧い", "たくみい", "khéo tay; tài giỏi"), ("巧み", "たくみ", "khéo léo; thông minh; lanh lợi")]},
    "朽": {"viet": "HỦ", "meaning_vi": "Gỗ mục, phàm vật gì thối nát đều gọi là hủ cả. Như hủ mộc [朽木] gỗ mục.", "meo": "", "reading": "く.ちる", "vocab": [("不朽", "ふきゅう", "bất hủ"), ("朽廃", "きゅうはい", "tình trạng suy tàn")]},
    "誇": {"viet": "KHOA, KHỎA", "meaning_vi": "Khoe khoang.", "meo": "", "reading": "ほこ.る", "vocab": [("誇り", "ほこり", "niềm tự hào; niềm kiêu hãnh ."), ("誇る", "ほこる", "tự hào; kiêu hãnh; tự cao; kiêu ngạo .")]},
    "袴": {"viet": "KHỐ", "meaning_vi": "Cái khố, quần đùi. Tục dùng như chữ khố [褲].", "meo": "", "reading": "はかま ずぼん", "vocab": [("袴", "はかま", "áo kimono của nam giới")]},
    "顎": {"viet": "NGẠC", "meaning_vi": "Cái xương gò má, (quyền) xương quai hàm gọi là hạ ngạc [下顎].", "meo": "", "reading": "あご あぎと", "vocab": [("顎", "あご", "cái cằm"), ("上顎", "うわあご", "hàm trên")]},
    "愕": {"viet": "NGẠC", "meaning_vi": "Hớt hải, kinh ngạc. Tả cái dáng sợ hãi cuống cuồng. Như ngạc nhiên [愕然].", "meo": "", "reading": "おどろ.く", "vocab": [("愕然", "がくぜん", "sự ngạc nhiên"), ("驚愕", "きょうがく", "sự ngạc nhiên")]},
    "鰐": {"viet": "NGẠC", "meaning_vi": "Cũng như chữ ngạc [鱷].", "meo": "", "reading": "わに", "vocab": [("鰐", "わに", "cá sấu"), ("内鰐", "うちわに", "có chân vòng kiềng")]},
    "妖": {"viet": "YÊU", "meaning_vi": "yêu quái, ma mị", "meo": "Chữ Nữ (女) gặp Rốt (夭) thành Yêu quái.", "reading": "よう", "vocab": [("妖精", "ようせい", "tiên nữ, tinh linh"), ("妖怪", "ようかい", "yêu quái, ma quỷ")]},
    "沃": {"viet": "ỐC", "meaning_vi": "Rót vào, bón tưới, lấy chất lỏng đặc nồng béo rót vào gọi là ốc.", "meo": "", "reading": "そそ.ぐ", "vocab": [("沃化", "ようか", "sự bôi iôt"), ("沃地", "よくち", "(địa lý")]},
    "添": {"viet": "THIÊM", "meaning_vi": "Thêm, thêm lên. Như cẩm thượng thiêm hoa [錦上添花] trên gấm thêm hoa, ý nói đã đẹp lại đẹp thêm. Nguyễn Trãi [阮廌] : Độ đầu xuân thảo lục như yên, Xuân vũ thiêm lai thủy phách thiên [渡頭春草綠如煙, 春雨添來水拍天] (Trại đầu xuân độ [寨頭春渡]) Ở bến đò đầu trại, cỏ xuân xanh như khói, Lại thêm mưa xuân, nước vỗ vào nền trời.", "meo": "", "reading": "そ.える そ.う も.える も.う", "vocab": [("添う", "そう", "đi cùng; theo"), ("付添", "つきそい", "tham dự")]},
    "呑": {"viet": "THÔN", "meaning_vi": "Nuốt.", "meo": "", "reading": "の.む", "vocab": [("呑む", "のむ", "đồ uống"), ("併呑", "へいどん", "sự phụ vào; sự thêm vào")]},
    "狂": {"viet": "CUỒNG", "meaning_vi": "Bệnh hóa rồ. Như cuồng nhân [狂人] người rồ, cuồng khuyển [狂犬] chó dại.", "meo": "", "reading": "くる.う くる.おしい くるお.しい", "vocab": [("狂い", "くるい", "sự trệch"), ("狂う", "くるう", "điên; điên khùng; mất trí; hỏng hóc; trục trặc")]},
    "旺": {"viet": "VƯỢNG", "meaning_vi": "Sáng sủa, tốt đẹp. Phàm vật gì mới thịnh gọi là vượng. Như hưng vượng [興旺] tốt đẹp mạnh mẽ, thịnh vượng [盛旺] ngày thêm tốt đẹp hơn, v.v.", "meo": "", "reading": "かがや.き うつくし.い さかん", "vocab": []},
    "班": {"viet": "BAN", "meaning_vi": "Ban phát, chia cho.", "meo": "", "reading": "ハン", "vocab": [("班", "はん", "kíp; đội; nhóm"), ("班次", "はんじ", "quyền được trước")]},
    "斑": {"viet": "BAN", "meaning_vi": "Lang lổ. Nguyễn Trãi [阮廌] : Bi khắc tiển hoa ban  [碑刻蘚花斑] (Dục Thúy sơn [浴翠山]) Bia khắc đã lốm đốm rêu.", "meo": "", "reading": "ふ まだら", "vocab": [("斑", "ぶち", "vết đốm; vết lốm đốm ."), ("一斑", "いちむら", "nửa")]},
    "栓": {"viet": "XUYÊN", "meaning_vi": "Cái then cửa, cái chốt cửa. Tục gọi cái nút chai là xuyên.", "meo": "", "reading": "セン", "vocab": [("栓", "せん", "nút ."), ("共栓", "きょうせん", "người làm ngừng")]},
    "詮": {"viet": "THUYÊN", "meaning_vi": "Đủ, giải thích kỹ càng, nói đủ cả sự lẽ gọi là thuyên. Như thuyên giải [詮解] giải rõ nghĩa lý. Lại như phân tích những lẽ khó khăn mà tìm tới nghĩa nhất định gọi là chân thuyên [真詮] chân lý của mọi sự, sự thật.", "meo": "", "reading": "せん.ずる かい あき.らか", "vocab": [("詮ない", "せんない", "không thể tránh được"), ("詮索", "せんさく", "sự điều tra nghiên cứu")]},
    "厄": {"viet": "ÁCH, NGỎA", "meaning_vi": "Cũng như chữ ách [阨] nghĩa là khốn ách. Như khổ ách [苦厄] khổ sở.", "meo": "", "reading": "ヤク", "vocab": [("厄", "やく", "điều bất hạnh"), ("厄介", "やっかい", "phiền hà; rắc rối; gây lo âu")]},
    "範": {"viet": "PHẠM", "meaning_vi": "Phép, khuôn mẫu. Đàn bà có đức hạnh trinh thục gọi là khuê phạm [閨範].", "meo": "", "reading": "ハン", "vocab": [("範", "はん", "thí dụ"), ("範例", "はんれい", "ví dụ .")]},
    "氾": {"viet": "PHIẾM", "meaning_vi": "Giàn giụa.", "meo": "", "reading": "ひろ.がる", "vocab": [("氾濫", "はんらん", "sự tràn lan ."), ("氾濫する", "はんらん", "tràn lan .")]},
    "怨": {"viet": "OÁN", "meaning_vi": "Oán giận.", "meo": "", "reading": "うら.む うらみ うら.めしい", "vocab": [("怨み", "うらみ", "oán thù ."), ("怨む", "うらむ", "hiềm")]},
    "椀": {"viet": "OẢN", "meaning_vi": "Cái bát nhỏ, cùng một nghĩa với chữ oản [盔]. Tục quen viết là oản [碗].", "meo": "", "reading": "はち", "vocab": [("椀", "わん", "bát Nhật; bát gỗ ."), ("椀ぐ", "もぐ", "hái; vặt .")]},
    "碗": {"viet": "OẢN", "meaning_vi": "Tục dùng như chữ oản [盌].", "meo": "", "reading": "こばち", "vocab": [("碗", "わん", "bát"), ("お碗", "おわん", "bát đựng nước tương; bát; chén (theo cách gọi của người Nam Bộ)")]},
    "誉": {"viet": "DỰ", "meaning_vi": "Giản thể của chữ [譽].", "meo": "", "reading": "ほま.れ ほ.める", "vocab": [("誉れ", "ほまれ", "danh dự; thanh danh"), ("名誉", "めいよ", "có danh dự")]},
    "逮": {"viet": "ĐÃI", "meaning_vi": "Kịp. Như Luận ngữ [論語] nói sỉ cung chi bất đãi [恥躬之不逮] (Lý nhân [里仁]) hổ mình không theo kịp.", "meo": "", "reading": "タイ", "vocab": [("逮捕", "たいほ", "bắt bỏ tù"), ("逮捕する", "たいほ", "bắt; tóm; chặn lại; bắt giữ")]},
    "隷": {"viet": "LỆ", "meaning_vi": "Cũng như chữ lệ [隸].", "meo": "", "reading": "したが.う しもべ", "vocab": [("奴隷", "どれい", "nô lệ; người hầu"), ("隷属", "れいぞく", "sự lệ thuộc .")]},
    "粛": {"viet": "TÚC", "meaning_vi": "Tục dùng như chữ túc [肅].", "meo": "", "reading": "つつし.む", "vocab": [("粛党", "しゅくとう", "sự chỉnh lý đảng ."), ("厳粛", "げんしゅく", "nghiêm trang; nghiêm nghị; uy nghiêm; trang trọng")]},
    "繍": {"viet": "", "meaning_vi": "sew, figured cloth", "meo": "", "reading": "ぬいとり", "vocab": [("刺繍", "ししゅう", "thêu dệt ."), ("刺繍する", "ししゅうする", "thêu .")]},
    "庸": {"viet": "DONG, DUNG", "meaning_vi": "Dùng. Như đăng dong [登庸] cất lên ngôi mà dùng. Có khi dùng làm tiếng trợ ngữ. Như vô dong như thử [無庸如此] không cần dùng như thế.", "meo": "", "reading": "ヨウ", "vocab": [("中庸", "ちゅうよう", "ôn hoà; điều độ"), ("凡庸", "ぼんよう", "sự tầm thường; sự xoàng xĩnh")]},
    "唐": {"viet": "ĐƯỜNG", "meaning_vi": "Nói khoác, nói không có đầu mối gì gọi là hoang đường [荒唐], không chăm nghề nghiệp chính đính cũng gọi là hoang đường.", "meo": "", "reading": "から", "vocab": [("唐", "とう", "nhà Đường; đời Đường"), ("唐人", "とうじん", "Trung quốc")]},
    "避": {"viet": "TỴ", "meaning_vi": "tránh né", "meo": "Tránh xa cái XÁC (尸) bằng con DAO (辛) đi!", "reading": "ひ", "vocab": [("避難", "ひなん", "lánh nạn, tị nạn"), ("避ける", "さける", "tránh, né tránh")]},
    "璧": {"viet": "BÍCH", "meaning_vi": "Ngọc bích. Nguyễn Du [阮攸] : Hoàng kim bách dật, bích bách song [黃金百鎰璧百雙] (Tô Tần đình [蘇秦亭]) Hoàng kim trăm dật, ngọc bích trăm đôi.", "meo": "", "reading": "たま", "vocab": [("完璧", "かんぺき", "hoàn mỹ; thập toàn; toàn diện; chuẩn"), ("完璧さ", "かんぺきさ", "sự hoàn thành")]},
    "癖": {"viet": "PHÍCH, TÍCH", "meaning_vi": "Bệnh hòn trong bụng.", "meo": "", "reading": "くせ くせ.に", "vocab": [("癖", "くせ", "thói hư; tật xấu"), ("一癖", "ひとくせ", "nét")]},
    "臨": {"viet": "LÂM, LẤM", "meaning_vi": "Ở trên soi xuống. Như giám lâm [監臨] soi xét, đăng lâm [登臨] ngắm nghía. Đỗ Phủ [杜甫] : Hoa cận cao lâu thương khách tâm, Vạn phương đa nạn thử đăng lâm [花近高樓傷客心，萬方多難此登臨] (Đăng lâu [登樓]) Hoa ở gần lầu cao làm đau lòng khách, (Trong khi) ở muôn phương nhiều nạn, ta lên lầu này ngắm ra xa.", "meo": "", "reading": "のぞ.む", "vocab": [("臨む", "のぞむ", "tiến đến; tiếp cận"), ("再臨", "さいりん", "sự trở lại của Chúa Giê")]},
    "謳": {"viet": "ÂU", "meaning_vi": "Cất tiếng cùng hát, ngợi hát.", "meo": "", "reading": "うた.う", "vocab": [("謳う", "うたう", "chủ trương; tán thành; ủng hộ"), ("謳歌", "おうか", "sự tuyên dương")]},
    "癌": {"viet": "NHAM", "meaning_vi": "Một thứ nhọt mọc ở trong tạng phủ và ở ngoài, lồi lõm không đều, rắn chắc mà đau, ở dạ dày gọi là vị nham [胃癌], ở vú gọi là nhũ nham [乳癌].", "meo": "", "reading": "ガン", "vocab": [("癌", "がん", "bệnh ung thư"), ("乳癌", "にゅうがん", "bệnh ung thư vú")]},
    "操": {"viet": "THAO, THÁO", "meaning_vi": "Cầm, giữ. Như thao khoán [操券] cầm khoán.", "meo": "", "reading": "みさお あやつ.る", "vocab": [("操", "みさお", "danh dự"), ("操り", "あやつり", "sự vận dụng bằng tay")]},
    "繰": {"viet": "SÀO", "meaning_vi": "Cũng như chữ sào [繅].", "meo": "", "reading": "く.る", "vocab": [("繰る", "くる", "quay; quấn; cuộn; mở; lần; xe"), ("繰言", "くげん", "sự nhắc lại")]},
    "藻": {"viet": "TẢO", "meaning_vi": "Rong, rau bể, tên gọi tất cả các thứ cỏ mọc ở dưới nước.", "meo": "", "reading": "も", "vocab": [("藻", "も", "loài thực vật trong ao đầm sông hồ biển như bèo rong tảo"), ("海藻", "かいそう", "hải thảo")]},
    "奨": {"viet": "TƯỞNG", "meaning_vi": "Tưởng thưởng", "meo": "", "reading": "すす.める", "vocab": [("奨励", "しょうれい", "sự động viên; sự khích lệ; sự khuyến khích ."), ("勧奨", "かんしょう", "sự khuyến khích; khuyến khích")]},
    "醤": {"viet": "", "meaning_vi": "a kind of miso", "meo": "", "reading": "ひしお", "vocab": [("醤油", "しょうゆ", "xì dầu .")]},
    "弓": {"viet": "CUNG", "meaning_vi": "Cái cung.", "meo": "", "reading": "ゆみ", "vocab": [("弓", "ゆみ", "cái cung"), ("大弓", "だいきゅう", "cái cung")]},
    "窮": {"viet": "CÙNG", "meaning_vi": "Cùng cực, cái gì đến thế là hết nước đều gọi là cùng. Như bần cùng [貧窮] nghèo quá, khốn cùng [困窮] khốn khó quá, v.v. Luận ngữ [論語] : Quân tử cố cùng, tiểu nhân cùng tư lạm hĩ [君子固窮, 小人窮斯濫矣] (Vệ Linh Công [衛靈公]) Người quân tử có khi cùng khốn cũng là lẽ cố nhiên (thi cố giữ tư cách của mình); kẻ tiểu nhân khốn cùng thì phóng túng làm càn.", "meo": "", "reading": "きわ.める きわ.まる きわ.まり きわ.み", "vocab": [("窮乏", "きゅうぼう", "sự cùng khốn; sự túng quẫn; sự khốn cùng; sự túng thiếu; sự thiếu thốn; sự túng bấn; khốn cùng; túng thiếu; thiếu thống; túng bấn"), ("窮余", "きゅうよ", "đầu")]},
    "弔": {"viet": "ĐIẾU, ĐÍCH", "meaning_vi": "Viếng thăm, đến viếng người chết và hỏi thăm những sự không may của những người thân thích của kẻ chết gọi là điếu.", "meo": "", "reading": "とむら.う とぶら.う", "vocab": [("弔", "ちょう", "sự đau buồn"), ("弔い", "とむらい", "sự chôn cất")]},
    "剃": {"viet": "THẾ", "meaning_vi": "Cắt tóc. Cắt tóc đi tu gọi là thế phát [剃髮].", "meo": "", "reading": "まい そ.る す.る", "vocab": [("剃", "そ", "sự cạo râu"), ("剃る", "そる", "cạo")]},
    "梯": {"viet": "THÊ", "meaning_vi": "Cái thang. Như lâu thê [樓梯] thang lầu.", "meo": "", "reading": "はしご", "vocab": [("梯子", "はしこ", "thang gác"), ("梯子", "はしご", "cầu thang")]},
    "沸": {"viet": "PHÍ, PHẤT", "meaning_vi": "Sôi. Như phí thủy [沸水] nước sôi.", "meo": "", "reading": "わ.く わ.かす", "vocab": [("沸々", "にえ々", "trạng thái sắp sôi"), ("沸く", "わく", "sôi lên")]},
    "溺": {"viet": "NỊCH, NIỆU", "meaning_vi": "Chết đuối, chìm mất. Bị chìm ở trong nước gọi là nịch [溺].", "meo": "", "reading": "いばり おぼ.れる", "vocab": [("溺らす", "おぼらす", "chết đuối"), ("溺れる", "おぼれる", "chết đuối; chìm đắm; đắm chìm; ngất ngây; chìm ngập; ham mê; say mê vô độ")]},
    "鰯": {"viet": "", "meaning_vi": "sardine, (kokuji)", "meo": "", "reading": "いわし", "vocab": [("鰯", "いわし", "cá mòi"), ("赤鰯", "あかいわし", "Cá xacđin dầm giấm hoặc làm khô .")]},
    "那": {"viet": "NA, NẢ", "meaning_vi": "Nhiều. Như Kinh Thi có câu thụ phúc bất na [受福不那] chịu phúc chẳng nhiều.", "meo": "", "reading": "なに なんぞ いかん", "vocab": [("刹那", "せつな", "chốc"), ("旦那", "だんな", "ông chủ; ông chồng; ông xã")]},
    "邦": {"viet": "BANG", "meaning_vi": "Nước, nước lớn gọi là bang [邦], nước nhỏ gọi là quốc [國]. Nước láng giềng gọi là hữu bang [友邦].", "meo": "", "reading": "くに", "vocab": [("邦", "くに", "nước"), ("邦人", "ほうじん", "người bản quốc")]},
    "寿": {"viet": "THỌ", "meaning_vi": "Giản thể của chữ 壽", "meo": "", "reading": "ことぶき ことぶ.く ことほ.ぐ", "vocab": [("寿", "ことぶき", "lời chúc mừng sống lâu; Xin chúc thọ!"), ("寿く", "ことぶきく", "chúc mừng")]},
    "鋳": {"viet": "CHÚ", "meaning_vi": "Đúc, đúc quặng", "meo": "", "reading": "い.る", "vocab": [("鋳る", "いる", "đúc"), ("鋳型", "いがた", "khuôn đúc; khuôn .")]},
    "怯": {"viet": "KHIẾP", "meaning_vi": "Sợ khiếp, nhát.", "meo": "", "reading": "ひる.む おびえ.る おじる おび.える おそ.れる", "vocab": [("怯む", "ひるむ", "dao động"), ("卑怯", "ひきょう", "bần tiện")]},
    "却": {"viet": "KHƯỚC", "meaning_vi": "lùi lại, từ chối", "meo": "Bỏ rơi (却) cái chân (脚) trong quá khứ để tiến lên.", "reading": "kyaku", "vocab": [("冷却", "reikyaku", "làm lạnh"), ("返却", "henkyaku", "trả lại")]},
    "蓋": {"viet": "CÁI", "meaning_vi": "Che, trùm.", "meo": "", "reading": "ふた けだ.し おお.う かさ かこう", "vocab": [("蓋", "ふた", "cái nắp nồi"), ("蓋し", "けだし", "có lẽ")]},
    "弁": {"viet": "BIỆN, BIỀN, BÀN", "meaning_vi": "Cái mũ lớn đời xưa. Chia ra hai thứ mũ da và mũ tước, mũ da để quan võ dùng, mũ tước để quan văn dùng.", "meo": "", "reading": "かんむり わきま.える わ.ける はなびら あらそ.う", "vocab": [("弁", "べん", "có tài hùng biện"), ("代弁", "だいべん", "sự thay mặt người khác để phát ngôn")]},
    "弄": {"viet": "LỘNG", "meaning_vi": "Mân mê, ngắm nghía. Nay gọi sinh con trai là lộng chương [弄璋], sinh con gái là lộng ngõa [弄瓦].", "meo": "", "reading": "いじく.る ろう.する いじ.る ひねく.る たわむ.れる もてあそ.ぶ", "vocab": [("弄り", "いじり", "sự xen vào việc người khác"), ("弄る", "いじる", "chạm")]},
    "奔": {"viet": "BÔN", "meaning_vi": "Chạy vội. Như bôn trì [奔馳] rong ruổi.", "meo": "", "reading": "はし.る", "vocab": [("出奔", "しゅっぽん", "sự chạy trốn; chạy trốn ."), ("奔命", "ほんめい", "được mến chuộng")]},
    "升": {"viet": "THĂNG", "meaning_vi": "Thưng, mười lẻ là một thưng.", "meo": "", "reading": "ます", "vocab": [("升", "ます", "thăng"), ("一升", "いっしょう", "một thăng")]},
    "尿": {"viet": "NIỆU", "meaning_vi": "Nước đái (nước giải).", "meo": "", "reading": "ニョウ", "vocab": [("尿", "にょう", "nước đái"), ("利尿", "りにょう", "lợi tiểu")]},
    "尾": {"viet": "VĨ", "meaning_vi": "Đuôi.", "meo": "", "reading": "お", "vocab": [("尾", "お", "cái đuôi"), ("大尾", "たいび", "giới hạn")]},
    "尻": {"viet": "KHÀO, CỪU", "meaning_vi": "Xương cùng đít. Ta quen đọc là chữ cừu.", "meo": "", "reading": "しり", "vocab": [("尻", "しり", "mông; cái mông; đằng sau ."), ("お尻", "おしり", "mông đít; hậu môn; đít")]},
    "尼": {"viet": "NI, NỆ, NẶC, NẬT", "meaning_vi": "Ni khâu [尼丘] núi Ni-khâu, đức Khổng-mẫu (Nhan thị) cầu nguyện ở núi ấy sinh ra đức Khổng-tử, nên mới đặt tên ngài là Khâu.", "meo": "", "reading": "あま", "vocab": [("尼", "あま", "bà xơ; ma xơ"), ("尼僧", "にそう", "nữ tu; cô đồng")]},
    "炉": {"viet": "LÔ", "meaning_vi": "Giản thể của chữ 爐", "meo": "", "reading": "いろり", "vocab": [("炉", "ろ", "lò"), ("炉床", "ろゆか", "nền lò sưởi")]},
    "啓": {"viet": "KHẢI", "meaning_vi": "Như chữ [啟].", "meo": "", "reading": "ひら.く さと.す", "vocab": [("啓く", "けいく", "làm sáng tỏ"), ("天啓", "てんけい", "đền thờ linh thiêng")]},
    "眉": {"viet": "MI", "meaning_vi": "Lông mày.", "meo": "", "reading": "まゆ", "vocab": [("眉", "まゆ", "lông mày ."), ("眉宇", "びう", "mày")]},
    "翻": {"viet": "PHIÊN", "meaning_vi": "Phiên phiên [翻翻] bay vùn vụt, bay.", "meo": "", "reading": "ひるがえ.る ひるがえ.す", "vocab": [("翻す", "ひるがえす", "bay phấp phới"), ("翻る", "ひるがえる", "bay phấp phới")]},
    "藩": {"viet": "PHIÊN, PHAN", "meaning_vi": "Bờ rào.", "meo": "", "reading": "ハン", "vocab": [("藩", "はん", "thái ấp; đất phong"), ("藩主", "はんしゅ", "lãnh chúa .")]},
    "審": {"viet": "THẨM", "meaning_vi": "Xét rõ, xét kĩ.", "meo": "", "reading": "つまび.らか つぶさ.に", "vocab": [("一審", "いちしん", "sự toàn tâm toàn ý"), ("不審", "ふしん", "không rõ ràng")]},
    "冥": {"viet": "MINH", "meaning_vi": "Chỗ mù mịt không có ánh sáng. Như minh trung [冥中] trong chốn u minh.", "meo": "", "reading": "くら.い", "vocab": [("冥", "めい", "tối"), ("冥々", "めいめい", "tối")]},
    "瞑": {"viet": "MINH, MIỄN", "meaning_vi": "Nhắm mắt. Người chết nhắm mắt gọi là minh mục [瞑目]. Chết không nhắm mắt gọi là tử bất minh mục [死不瞑目]. Quân nhược xả ngã nhi khứ, ngã tử bất minh mục hỹ [君若捨我而去，我死不瞑目矣] (Tam quốc diễn nghĩa [三國演義]) nếu ngươi bỏ ta mà đi, ta chết không nhắm mắt đâu.", "meo": "", "reading": "めい.する つぶ.る つむ.る くら.い", "vocab": [("瞑る", "つぶる", "nhắm mắt"), ("瞑る", "つむる", "nhắm mắt")]},
    "罷": {"viet": "BÃI, BÌ", "meaning_vi": "Nghỉ, thôi. Như bãi công [罷工] thôi không làm việc nữa, bãi thị [罷市] bỏ không họp chợ nữa.", "meo": "", "reading": "まか.り- や.める", "vocab": [("罷免", "ひめん", "sự thải hồi; sự đuổi đi; sự sa thải ."), ("罷官", "まかかん", "sự trú đông")]},
    "熊": {"viet": "HÙNG", "meaning_vi": "Con gấu.", "meo": "", "reading": "くま", "vocab": [("熊", "くま", "gấu; con gấu"), ("熊手", "くまで", "cào; cái cào")]},
    "属": {"viet": "CHÚC, THUỘC, CHÚ", "meaning_vi": "Cũng như chữ chú [屬].", "meo": "", "reading": "さかん つく やから", "vocab": [("亜属", "あぞく", "phân nhóm"), ("付属", "ふぞく", "phụ thuộc")]},
    "嘱": {"viet": "CHÚC", "meaning_vi": "Giản thể của chữ 囑", "meo": "", "reading": "しょく.する たの.む", "vocab": [("委嘱", "いしょく", "sự dặn dò; sự ủy thác"), ("嘱望", "しょくぼう", "sự kỳ vọng; sự hy vọng; kỳ vọng; hy vọng .")]},
    "呉": {"viet": "NGÔ", "meaning_vi": "Nước Ngô, họ Ngô, đất Ngô.", "meo": "", "reading": "く.れる くれ", "vocab": [("呉", "ご", "sự làm vì ai; việc làm cho ai ."), ("呉れる", "くれる", "cho; tặng")]},
    "娯": {"viet": "NGU", "meaning_vi": "Vui chơi, ngu lạc", "meo": "", "reading": "ゴ", "vocab": [("娯楽", "ごらく", "du hý"), ("娯楽場", "ごらくじょう", "nơi giải trí .")]},
    "虞": {"viet": "NGU", "meaning_vi": "Đo đắn, dự liệu.", "meo": "", "reading": "おそれ おもんぱか.る はか.る うれ.える あざむ.く あやま.る のぞ.む たの.しむ", "vocab": [("虞", "おそれ", "sự sợ"), ("危虞", "きく", "sợ hãi; những nỗi lo âu; lo âu")]},
    "汰": {"viet": "THÁI, THẢI", "meaning_vi": "Quá. Như xa thái [奢汰] xa xỉ quá.", "meo": "", "reading": "おご.る にご.る よな.げる", "vocab": [("沙汰", "さた", "việc"), ("淘汰", "とうた", "Sự chọn lọc (tự nhiên)")]},
    "駄": {"viet": "ĐÀ", "meaning_vi": "Thồ hàng.", "meo": "", "reading": "ダ タ", "vocab": [("一駄", "いちだ", "cú"), ("下駄", "げた", "guốc")]},
    "鰹": {"viet": "", "meaning_vi": "bonito", "meo": "", "reading": "かつお", "vocab": [("鰹", "かつお", "cá giác .")]},
    "腎": {"viet": "THẬN", "meaning_vi": "Bồ dục, quả cật.", "meo": "", "reading": "ジン", "vocab": [("腎炎", "じんえん", "viêm thận"), ("腎症", "じんしょう", "bệnh thận .")]},
    "臼": {"viet": "CỮU", "meaning_vi": "Cái cối, làm bằng gỗ hay bằng đá để giã các thứ. Thạch cữu [石臼] cối đá.", "meo": "", "reading": "うす うすづ.く", "vocab": [("臼", "うす", "cối ."), ("臼曇", "うすぐもり", "trời mây hơi đục; trời nhiều mây")]},
    "潟": {"viet": "TÍCH", "meaning_vi": "Đất mặn, đất có chất muối.", "meo": "", "reading": "かた -がた", "vocab": [("潟", "かた", "phá"), ("干潟", "ひがた", "bãi cát lộ ra sau khi thủy triều xuống .")]},
    "毀": {"viet": "HỦY", "meaning_vi": "Phá, nát. Như hủy hoại [毀壞] phá hư, hủy diệt [毀滅] phá hỏng, làm mất đi.", "meo": "", "reading": "こぼ.つ こわ.す こぼ.れる こわ.れる そし.る やぶ.る", "vocab": [("毀つ", "こぼつ", "phá"), ("毀傷", "きしょう", "sự làm hại")]},
    "睨": {"viet": "NGHỄ", "meaning_vi": "Nghé trông, liếc.", "meo": "", "reading": "にら.む にら.み", "vocab": [("睨み", "にらみ", "ánh sáng"), ("睨む", "にらむ", "liếc; lườm")]},
    "鼠": {"viet": "THỬ", "meaning_vi": "Con chuột.", "meo": "", "reading": "ねずみ ねず", "vocab": [("鼠", "ねずみ", "con chuột; chuột"), ("地鼠", "じねずみ", "người đàn bà đanh đá")]},
    "興": {"viet": "HƯNG, HỨNG", "meaning_vi": "Dậy. Như túc hưng dạ mị [夙興夜寐] thức khuya dậy sớm.", "meo": "", "reading": "おこ.る おこ.す", "vocab": [("興", "きょう", "sự thưởng thức"), ("興", "こう", "hứng; sự hứng thú; sự hứng khởi")]},
    "輿": {"viet": "DƯ", "meaning_vi": "Cái xe tải đồ. Cái kiệu khiêng bằng đòn trên vai gọi là kiên dư [肩輿].", "meo": "", "reading": "かご こし", "vocab": [("輿", "こし", "kiệu"), ("輿地", "よち", "đất")]},
    "挟": {"viet": "HIỆP, TIỆP", "meaning_vi": "Giản thể của chữ 挾", "meo": "", "reading": "はさ.む はさ.まる わきばさ.む さしはさ.む", "vocab": [("挟む", "はさむ", "kẹp vào; chèn vào"), ("挟まる", "はさまる", "kẹp; kẹt vào giữa")]},
    "峡": {"viet": "HẠP, GIÁP", "meaning_vi": "Giản thể của chữ 峽", "meo": "", "reading": "はざま", "vocab": [("地峡", "ちきょう", "eo đất"), ("山峡", "さんきょう", "hẻm núi; khe núi")]},
    "頬": {"viet": "GIÁP", "meaning_vi": "gò má", "meo": "", "reading": "ほお ほほ", "vocab": [("頬", "ほお", "má"), ("頬", "ほほ", "má")]},
    "為": {"viet": "VI, VỊ", "meaning_vi": "Dùng như chữ vi [爲].", "meo": "", "reading": "ため な.る な.す す.る たり つく.る なり", "vocab": [("為", "ため", "bởi vì; mục đích là; vì; cho; vị"), ("為す", "なす", "làm; hành động (kính ngữ)")]},
    "偽": {"viet": "NGỤY", "meaning_vi": "Như chữ ngụy [僞].", "meo": "", "reading": "いつわ.る にせ いつわ.り", "vocab": [("偽", "にせ", "sự bắt chước; sự giả"), ("偽り", "いつわり", "sự nói dối")]},
    "曹": {"viet": "TÀO", "meaning_vi": "bộ phận, lớp, Tào (họ)", "meo": "Ba (三) ông mặt trời (日) mồm (口) kêu gào vì bị xếp Tào (vào bộ phận).", "reading": "そう", "vocab": [("曹長", "そうちょう", "Trung sĩ"), ("法曹", "ほうそう", "Giới luật sư, tư pháp")]},
    "槽": {"viet": "TÀO", "meaning_vi": "Cái máng cho giống muông ăn.", "meo": "", "reading": "ふね", "vocab": [("歯槽", "しそう", "/æl'viəlai/"), ("水槽", "すいそう", "thùng chứa nước; bể chứa nước; két nước .")]},
    "遭": {"viet": "TAO", "meaning_vi": "Gặp, vô ý mà gặp nhau gọi là tao. Như tao phùng ý ngoại [遭逢意外] gặp gỡ ý không ngờ tới.", "meo": "", "reading": "あ.う あ.わせる", "vocab": [("遭う", "あう", "gặp; gặp phải"), ("遭遇", "そうぐう", "cuộc chạm trán; sự bắt gặp thình lình; sự bắt gặp .")]},
    "漕": {"viet": "TÀO", "meaning_vi": "Vận tải đường nước. Như vận lương thực đi đường nước gọi là tào mễ [演米].", "meo": "", "reading": "こ.ぐ はこ.ぶ", "vocab": [("漕ぐ", "こぐ", "chèo thuyền; chèo; lái"), ("回漕", "かいそう", "sự xếp hàng xuống tàu; sự chở hàng bằng tàu")]},
    "后": {"viet": "HẬU, HẤU", "meaning_vi": "Vua, đời xưa gọi các chư hầu là quần hậu [羣后].", "meo": "", "reading": "きさき", "vocab": [("后", "きさき", "Hoàng hậu; nữ hoàng ."), ("午后", "ごご", "buổi chiều")]},
    "垢": {"viet": "CẤU", "meaning_vi": "Cáu bẩn. Như khứ cấu [去垢] làm hết dơ bẩn.", "meo": "", "reading": "あか はじ", "vocab": [("垢", "あか", "cặn; cáu bẩn (ở trong nước)"), ("歯垢", "しこう", "bựa răng .")]},
    "灸": {"viet": "CỨU", "meaning_vi": "Cứu, lấy ngải cứu châm lửa đốt vào các huyệt để chữa bệnh gọi là cứu.", "meo": "", "reading": "やいと", "vocab": [("温灸", "おんきゅう", "lương hưu"), ("針灸", "しんきゅう", "pháp châm cứu .")]},
    "畝": {"viet": "MẪU", "meaning_vi": "Mẫu, mười sào là một mẫu (3600 thước vuông tây là một mẫu).", "meo": "", "reading": "せ うね", "vocab": [("畝", "うね", "luống cây; luống"), ("畝", "せ", "100 mét vuông .")]},
    "凄": {"viet": "THÊ", "meaning_vi": "Tục dùng như chữ [淒].", "meo": "", "reading": "さむ.い すご.い すさ.まじい", "vocab": [("凄", "すご", "doạ"), ("凄い", "すごい", "kinh khủng; khủng khiếp")]},
    "沿": {"viet": "DUYÊN", "meaning_vi": "Ven. Như duyên thủy nhi hạ [沿水而下] ven nước mà xuống.", "meo": "", "reading": "そ.う -ぞ.い", "vocab": [("沿い", "ぞい", "dọc theo; men theo"), ("沿う", "そう", "chạy dài; chạy theo suốt; dọc theo; men theo")]},
    "鉛": {"viet": "DUYÊN, DIÊN", "meaning_vi": "Chì, một loài kim giống như thiếc mà mềm (Plumbum, Pb). Cho giấm vào nấu, có thể chế ra phấn. Các nhà tu đạo ngày xưa dùng để luyện thuốc.", "meo": "", "reading": "なまり", "vocab": [("鉛", "なまり", "chì"), ("亜鉛", "あえん", "kẽm")]},
    "朕": {"viet": "TRẪM", "meaning_vi": "Ta đây, tiếng dùng của kẻ tôn quý. Như vua tự nói mình thì tự xưng là trẫm. Tô Tuân [蘇洵] : Trẫm chí tự định [朕志自定] (Trương Ích Châu họa tượng kí [張益州畫像記]) Ý trẫm đã định.", "meo": "", "reading": "チン", "vocab": [("朕", "ちん", "trẫm (tiếng xưng của nhà vua)"), ("朕思うに", "ちんおもうに", "trẫm (tiếng xưng của nhà vua) .")]},
    "丹": {"viet": "ĐAN", "meaning_vi": "Đan sa [丹砂], tức là chu sa [朱砂] đời xưa dùng làm thuốc mùi, đều gọi tắt là đan [丹]. Như nói về sự vẽ thì gọi là đan thanh [丹青], nói về sự xét sửa lại sách vở gọi là đan duyên [丹鉛], đan hoàng [丹黄], v.v.", "meo": "", "reading": "に", "vocab": [("丹", "に", "đất đỏ (ngày xưa thường dùng để nhuộm); màu đỏ đất"), ("丹前", "たんぜん", "một loại áo bông dày")]},
    "皇": {"viet": "HOÀNG", "meaning_vi": "To lớn, tiếng gọi tôn kính. Như hoàng tổ [皇祖] ông, hoàng khảo [皇考] cha, v.v.", "meo": "", "reading": "コウ オウ", "vocab": [("人皇", "にんのう", "Hoàng đế ."), ("皇位", "こうい", "ngôi hoàng đế; vị trí hoàng đế .")]},
    "呈": {"viet": "TRÌNH", "meaning_vi": "Bảo, tỏ ra.", "meo": "", "reading": "テイ", "vocab": [("呈上", "ていじょう", "sự bày ra"), ("呈出", "ていしゅつ", "pri'zent/")]},
    "聖": {"viet": "THÁNH", "meaning_vi": "Thánh, tu dưỡng nhân cách tới cõi cùng cực gọi là thánh. Như siêu phàm nhập thánh [超凡入聖] vượt khỏi cái tính phàm trần mà vào cõi thánh.", "meo": "", "reading": "ひじり", "vocab": [("聖人", "せいじん", "thánh"), ("聖代", "せいだい", "rất quan trọng")]},
    "唱": {"viet": "XƯỚNG", "meaning_vi": "Hát, ca. Nguyễn Trãi [阮廌] : Ngư ca tam xướng yên hồ khoát [漁歌三唱烟湖闊] (Chu trung ngẫu thành [舟中偶成]) Chài ca mấy khúc, khói hồ rộng mênh mông.", "meo": "", "reading": "とな.える", "vocab": [("主唱", "しゅしょう", "chủ trương; đề xướng ."), ("伝唱", "でんしょう", "Truyền thống .")]},
    "娼": {"viet": "XƯỚNG", "meaning_vi": "Con hát. Cũng như chữ xướng [倡].", "meo": "", "reading": "あそびめ", "vocab": [("公娼", "こうしょう", "Gái mại dâm có giấy phép hành nghề"), ("娼妓", "しょうぎ", "to prostitute oneself làm đĩ")]},
    "殖": {"viet": "THỰC", "meaning_vi": "Sinh. Như phồn thực [蕃殖] sinh sôi, nẩy nở.", "meo": "", "reading": "ふ.える ふ.やす", "vocab": [("利殖", "りしょく", "sự làm giàu; sự tích của"), ("増殖", "ぞうしょく", "sự tăng lên; sự sinh sản; sự nhân lên")]},
    "踏": {"viet": "ĐẠP", "meaning_vi": "Chân sát xuống đất. Làm việc vững chãi không mạo hiểm gọi là cước đạp thực địa [腳踏實地].", "meo": "", "reading": "ふ.む ふ.まえる", "vocab": [("踏む", "ふむ", "dẫm lên; trải qua"), ("踏切", "ふみきり", "nơi chắn tàu .")]},
    "剥": {"viet": "BÁC", "meaning_vi": "bóc, lột, tước", "meo": "Vũ khí (刂) cắt (刂) cây cối (木) ở trên và dưới (彔) để bóc vỏ.", "reading": "は.ぐ", "vocab": [("剥ぐ", "はぐ", "bóc, lột"), ("剥製", "はくせい", "đồ nhồi (động vật)")]},
    "禄": {"viet": "LỘC", "meaning_vi": "Cũng như chữ lộc [祿].", "meo": "", "reading": "さいわ.い ふち", "vocab": [("貫禄", "かんろく", "sự có mặt")]},
    "盾": {"viet": "THUẪN", "meaning_vi": "Cái mộc để đỡ tên mác.", "meo": "", "reading": "たて", "vocab": [("盾", "たて", "cái khiên; lá chắn; tấm mộc"), ("矛盾", "むじゅん", "mâu thuẫn")]},
    "循": {"viet": "TUẦN", "meaning_vi": "Noi, tuân theo. Như tuần pháp [循法] noi theo phép, tuần lý [循理] noi lẽ. Quan lại thuần lương gọi là tuần lại [循吏].", "meo": "", "reading": "ジュン", "vocab": [("因循", "いんじゅん", "sự do dự"), ("循環", "じゅんかん", "sự tuần hoàn; tuần hoàn .")]},
    "甲": {"viet": "GIÁP", "meaning_vi": "Can Giáp, một can đầu trong mười can. Ngày xưa lấy mười can kể lần lượt, cho nên cái gì hơn hết cả đều gọi là giáp. Như phú giáp nhất hương [富甲一鄉] giầu nhất một làng.", "meo": "", "reading": "きのえ", "vocab": [("甲", "かぶと", "vỏ; bao; mai"), ("甲乙", "こうおつ", "sự so sánh; sự tương tự giữa hai người")]},
    "岬": {"viet": "GIÁP", "meaning_vi": "Vệ núi. Giữa khoảng hai quả núi gọi là giáp. Núi thè vào bể cũng gọi là giáp.", "meo": "", "reading": "みさき", "vocab": [("岬", "みさき", "mũi đất ."), ("岬角", "こうかく", "mũi đất; doi đất; chỗ lồi lên; chỗ lồi")]},
    "撫": {"viet": "PHỦ, MÔ", "meaning_vi": "Yên ủi, phủ dụ. Như trấn phủ [鎮撫] đóng quân để giữ cho dân được yên, chiêu phủ [招撫] chiêu tập các kẻ lưu tán phản loạn về yên phận làm ăn, v.v.", "meo": "", "reading": "な.でる", "vocab": [("撫子", "なでしこ", "Hoa cẩm chướng ."), ("撫でる", "なでる", "xoa; sờ")]},
    "巫": {"viet": "VU", "meaning_vi": "Đồng cốt, kẻ cầu cúng cho người gọi là vu.", "meo": "", "reading": "みこ かんなぎ", "vocab": [("巫", "みこ", "người trung gian"), ("巫女", "みこ", "người trung gian")]},
    "挫": {"viet": "TỎA", "meaning_vi": "Gãy, chùn bước, thất bại", "meo": "Thổ ( đất ) chỉ để 2 người ngồi trên xe gặp nhau rồi giơ tay xin lỗi vì thất bại.", "reading": "くじ-く", "vocab": [("挫折", "ざせつ", "Thất bại, chán nản"), ("頓挫", "とんざ", "Đình trệ, bế tắc")]},
    "囃": {"viet": "TẠP", "meaning_vi": "play (music), accompany, beat time, banter, jeer, applaud", "meo": "", "reading": "はや.す", "vocab": [("囃子", "はやし", "dải"), ("持て囃す", "もてはやす", "đưa  đi thăm những cảnh lạ")]},
    "伐": {"viet": "PHẠT", "meaning_vi": "Đánh, đem binh đi đánh nước người gọi là phạt. Như chinh phạt [征伐] đem quân đi đánh nơi xa.", "meo": "", "reading": "き.る そむ.く う.つ", "vocab": [("伐", "ばつ", "sự tấn công; sự chinh phạt"), ("伐つ", "うつ", "đánh")]},
    "閥": {"viet": "PHIỆT", "meaning_vi": "Phiệt duyệt [閥閱] viết công trạng vào giấy hay tấm ván rồi nêu ra ngoài cửa, cửa bên trái gọi là phiệt, cửa bên phải gọi là duyệt. Sách Sử Ký [史記] nói : Nêu rõ thứ bực gọi là phiệt, tích số ngày lại gọi là duyệt. Vì thế nên gọi các nhà thế gia là phiệt duyệt [閥閱] hay thế duyệt [世閱].", "meo": "", "reading": "バツ", "vocab": [("閥", "ばつ", "bè đảng; phe cánh"), ("党閥", "とうばつ", "Đảng phái; bè cánh .")]},
    "箋": {"viet": "TIÊN", "meaning_vi": "Cuốn sách có chua ở trên và ở dưới để nêu rõ cái ý người xưa, hay lấy ý mình phán đoán khiến cho người ta dễ biết dễ nhớ gọi là tiên. Như sách của Trịnh Khang Thành [鄭康成] (127-200) chú thích Kinh Thi gọi là Trịnh tiên [鄭箋].", "meo": "", "reading": "ふだ", "vocab": [("付箋", "ふせん", "sắt bịt đầu"), ("便箋", "びんせん", "đồ văn phòng phẩm")]},
    "餓": {"viet": "NGẠ", "meaning_vi": "Đói quá. Nguyễn Trãi [阮廌] : Thú Dương ngạ tử bất thực túc [首陽餓死不食粟] (Côn sơn ca [崑山歌]) (Bá Di và Thúc Tề) ở núi Thú Dương chết đói, không chịu ăn thóc.", "meo": "", "reading": "う.える", "vocab": [("餓える", "うえる", "chết đói"), ("餓死", "うえじに", "sự chết đói; nạn chết đói")]},
    "鵞": {"viet": "", "meaning_vi": "goose", "meo": "", "reading": "ガ", "vocab": [("鵞鳥", "がちょう", "ngỗng"), ("天鵞絨", "びろうど", "nhung")]},
    "託": {"viet": "THÁC", "meaning_vi": "Nhờ. Gửi hình tích mình ở bên ngoài gọi là thác túc [託足].", "meo": "", "reading": "かこつ.ける かこ.つ かこ.つける", "vocab": [("託す", "たくす", "ủy thác"), ("託つ", "かこつ", "sự càu nhàu")]},
    "詫": {"viet": "THÁCH", "meaning_vi": "xin lỗi, tạ lỗi", "meo": "Lời (言) tạ lỗi khi cái gì đó THIẾU (宅) sót.", "reading": "わび", "vocab": [("詫びる", "わびる", "xin lỗi"), ("詫び状", "わびじょう", "thư xin lỗi")]},
    "托": {"viet": "THÁC", "meaning_vi": "Nâng, lấy tay mà nhấc vật gì lên gọi là thác.", "meo": "", "reading": "たく.する たの.む", "vocab": [("托する", "たくする", "giao")]},
    "羞": {"viet": "TU", "meaning_vi": "Dâng đồ ăn.", "meo": "", "reading": "はじ.る すすめ.る は.ずかしい", "vocab": [("含羞", "がんしゅう", "tính nhút nhát"), ("羞恥", "しゅうち", "tính nhút nhát")]},
    "錦": {"viet": "CẨM", "meaning_vi": "Gấm. Như ý cẩm [衣錦] mặc áo gấm, cũng chỉ người cao sang quyền quý.", "meo": "", "reading": "にしき", "vocab": [("錦", "にしき", "gấm ."), ("錦旗", "きんき", "niềm vui thích")]},
    "瓦": {"viet": "NGÕA", "meaning_vi": "Ngói.", "meo": "", "reading": "かわら ぐらむ", "vocab": [("瓦", "かわら", "ngói"), ("瓦斯", "がす", "khí")]},
    "餅": {"viet": "BÍNH", "meaning_vi": "Tục dùng như chữ bính [餠].", "meo": "", "reading": "もち もちい", "vocab": [("餅", "もち", "bánh dày ."), ("お餅", "おもち", "bánh dày")]},
    "併": {"viet": "TINH, TINH TỊNH, BÀNH", "meaning_vi": "Gộp lại, cùng nhau", "meo": "Hai người (亻亻) cùng đứng cạnh nhau (并) sẽ THỜ TINH (併) thần!", "reading": "へい", "vocab": [("併合", "へいごう", "Sáp nhập"), ("併記", "へいき", "Viết kèm")]},
    "塀": {"viet": "BIÊN", "meaning_vi": "Tường; rào", "meo": "", "reading": "ヘイ ベイ", "vocab": [("塀", "へい", "tường; vách; tường vây quanh"), ("土塀", "どべい", "tường bằng đất .")]},
    "暇": {"viet": "HẠ", "meaning_vi": "Nhàn rỗi.", "meo": "", "reading": "ひま いとま", "vocab": [("暇", "ひま", "thì giờ rỗi rãi"), ("暇な", "ひまな", "rảnh")]},
    "霞": {"viet": "HÀ", "meaning_vi": "Ráng. Trong khoảng trời không thâm thấp có khí mù, lại có bóng mặt trời xiên ngang thành các màu rực rỡ, thường thấy ở lúc mặt trời mới mọc hay mới lặn gọi là ráng. Vương Bột [王勃] : Lạc hà dữ cô vụ tề phi, thu thủy cộng trường thiên nhất sắc [落霞與孤鶩齊飛, 秋水共長天一色] (Đằng Vương Các tự [滕王閣序]) Ráng chiều với cánh vịt trời đơn chiếc cùng bay, nước thu trộn lẫn bầu trời dài một sắc.", "meo": "", "reading": "かすみ かす.む", "vocab": [("霞", "かすみ", "sương mù; màn che"), ("霞む", "かすむ", "mờ sương; che mờ; mờ; nhòa")]},
    "歪": {"viet": "OAI, OA", "meaning_vi": "Méo lệch. Ta quen đọc là chữ oa.", "meo": "", "reading": "いが.む いびつ ひず.む ゆが.む", "vocab": [("歪", "いびつ", "có hình trái xoan"), ("歪み", "ゆがみ", "sự căng")]},
    "甥": {"viet": "SANH", "meaning_vi": "Cháu ngoại, cháu gọi bằng cậu cũng gọi là sanh.", "meo": "", "reading": "おい むこ", "vocab": [("甥", "おい", "cháu trai")]},
    "湧": {"viet": "DŨNG", "meaning_vi": "Nước vọt ra. Tô Thức [蘇軾] : Phong khởi thủy dũng [風起水湧] (Hậu Xích Bích phú [後赤壁賦]) Gió nổi nước tung.", "meo": "", "reading": "わ.く", "vocab": [("湧く", "わく", "sôi sục"), ("湧き水", "わきみず", "nước nguồn")]},
    "溜": {"viet": "LỰU", "meaning_vi": "thu góp; để dành tiền.", "meo": "", "reading": "た.まる たま.る た.める したた.る たまり ため", "vocab": [("溜め", "ため", "hầm chứa phân"), ("溜り", "たまり", "tiền nợ lẽ ra phải trả trước đó; nợ còn khất lại")]},
    "瑠": {"viet": "LƯU", "meaning_vi": "Ngọc lưu ly", "meo": "", "reading": "ル リュウ", "vocab": [("瑠璃", "るり", "đá da trời"), ("瑠璃色", "るりしょく", "xanh da trời")]},
    "瘤": {"viet": "LỰU", "meaning_vi": "Cũng như chữ lựu [癅].", "meo": "", "reading": "こぶ", "vocab": [("瘤", "こぶ", "u; bướu; cục lồi lên"), ("瘤付き", "こぶつき", "kèm trẻ em .")]},
    "緯": {"viet": "VĨ", "meaning_vi": "Sợi ngang. Phàm thuộc về đường ngang đều gọi là vĩ. Xem chữ kinh [經].", "meo": "", "reading": "よこいと ぬき", "vocab": [("北緯", "ほくい", "bắc vĩ tuyến"), ("南緯", "なんい", "vỹ Nam; vỹ độ Nam")]},
    "柿": {"viet": "THỊ", "meaning_vi": "Cây thị, quả gọi thị tử [柿子] ăn được, gỗ dùng làm khí cụ.", "meo": "", "reading": "かき", "vocab": [("柿", "かき", "quả hồng ngâm; cây hồng ngâm; hồng ngâm; hồng (quả)"), ("熟柿", "じゅくし", "sự suy nghĩ cân nhắc kỹ")]},
    "肺": {"viet": "PHẾ", "meaning_vi": "Phổi, ở hai bên ngực, bên tả hai lá, bên hữu ba lá.", "meo": "", "reading": "ハイ", "vocab": [("肺", "はい", "phổi"), ("肺尖", "はいせん", "Đỉnh phổi (y)")]},
    "隔": {"viet": "CÁCH", "meaning_vi": "Ngăn cách, giữa khoảng hai cái gì mà lại có một cái ngăn cách ở giữa khiến cho không thông với nhau được gọi là cách. Như cách ngoa tao dưỡng [隔靴搔癢] cách giày gãi ngứa.", "meo": "", "reading": "へだ.てる へだ.たる", "vocab": [("隔て", "へだて", "sự chia ra"), ("隔世", "かくせい", "đánh thức")]},
    "塑": {"viet": "TỐ", "meaning_vi": "Đắp tượng.", "meo": "", "reading": "でく", "vocab": [("塑像", "そぞう", "tượng bằng đất hoặc đất nung"), ("可塑", "かそ", "chất dẻo")]},
    "遡": {"viet": "TỐ", "meaning_vi": "Ngoi lên. Ngược dòng bơi lên gọi là tố hồi [遡回]. Thuận dòng bơi xuống gọi là tố du [遡游].", "meo": "", "reading": "さかのぼ.る", "vocab": [("遡る", "さかのぼる", "đi ngược dòng"), ("遡上", "さかのぼうえ", "phản ứng")]},
    "俊": {"viet": "TUẤN", "meaning_vi": "Tài giỏi, tài trí hơn người gọi là tuấn. Phàm sự vật gì có tiếng hơn đời đều gọi là tuấn. Như tuấn kiệt [俊傑] người tài giỏi.", "meo": "", "reading": "シュン", "vocab": [("俊", "しゅん", "sự giỏi giang; sự ưu tú ."), ("俊傑", "しゅんけつ", "người tuấn kiệt; anh hùng .")]},
    "唆": {"viet": "TOA", "meaning_vi": "Xuýt làm, xúi làm. Như toa tụng [唆訟] xúi kiện.", "meo": "", "reading": "そそ.る そそのか.す", "vocab": [("唆し", "そそのかし", "sự xúi giục"), ("唆す", "そそのかす", "xúc")]},
    "酸": {"viet": "TOAN", "meaning_vi": "Chua.", "meo": "", "reading": "す.い", "vocab": [("酸", "さん", "a xít ."), ("乳酸", "にゅうさん", "Axít lactic (công thức hóa học là C3H6O3) .")]},
    "駿": {"viet": "TUẤN", "meaning_vi": "Con ngựa tốt. Như tuấn mã [駿馬].", "meo": "", "reading": "すぐ.れる", "vocab": [("駿才", "しゅんさい", "thiên tài")]},
    "竣": {"viet": "THUÂN, THUYÊN", "meaning_vi": "Thôi, xong việc. Như thuân sự [竣事] xong việc, thuân công [竣工] thành công, có khi đọc là chữ thuyên.", "meo": "", "reading": "わらわ わらべ おわ.る", "vocab": [("竣功", "しゅんこう", "sự hoàn thành"), ("竣工", "しゅんこう", "sự hoàn thành")]},
    "棺": {"viet": "QUAN, QUÁN", "meaning_vi": "Cái áo quan. Như nhập quan [入棺] bỏ xác người chết vào hòm, cái quan luận định [蓋棺論定] đậy nắp hòm mới khen chê hay dở.", "meo": "", "reading": "カン", "vocab": [("棺", "かん", "áo quan"), ("入棺", "にゅうかん", "Sự nhập quan (cho vào áo quan) .")]},
    "遣": {"viet": "KHIỂN, KHÁN", "meaning_vi": "Phân phát đi. Như khiển tán [遣散] phân phát đi hết.", "meo": "", "reading": "つか.う -つか.い -づか.い つか.わす や.る", "vocab": [("遣い", "つかい", "việc vắt (đưa thư"), ("遣う", "つかう", "cho; tặng; gửi đi")]},
    "槌": {"viet": "CHÙY", "meaning_vi": "Cái vồ lớn.", "meo": "", "reading": "つち", "vocab": [("槌", "つち", "búa ."), ("小槌", "こづち", "(từ Mỹ")]},
    "埠": {"viet": "PHỤ", "meaning_vi": "Bến đỗ. Chỗ thuyền buôn đỗ xếp các hàng hóa.", "meo": "", "reading": "つか はとば", "vocab": [("埠頭", "ふとう", "bến cảng"), ("埠頭税", "ふとうぜい", "phí cầu cảng .")]},
    "帥": {"viet": "SUẤT, SÚY", "meaning_vi": "Thống suất. Như suất sư [帥師] thống suất cả cánh quân đi.", "meo": "", "reading": "スイ", "vocab": [("元帥", "げんすい", "nguyên soái; thống chế; chủ soái; đô đốc"), ("将帥", "しょうすい", "người điều khiển")]},
    "獅": {"viet": "SƯ", "meaning_vi": "Con sư tử.", "meo": "", "reading": "しし", "vocab": [("獅子", "しし", "sư tử ."), ("獅子吼", "ししく", "bài diễn thuyết")]},
    "旨": {"viet": "CHỈ", "meaning_vi": "Ngon. Như chỉ tửu [旨酒] rượu ngon, cam chỉ [甘旨] ngon ngọt, v.v.", "meo": "", "reading": "むね うま.い", "vocab": [("旨", "むね", "chân lý"), ("旨い", "うまい", "thơm tho")]},
    "詣": {"viet": "NGHỆ", "meaning_vi": "Đến, đến thẳng tận nơi gọi là nghệ. Như xu nghệ [趨詣] đến thăm tận nơi.", "meo": "", "reading": "けい.する まい.る いた.る もう.でる", "vocab": [("参詣", "さんけい", "cuộc hành hương"), ("初詣で", "はつもうで", "việc đi lễ đền chùa ngày đầu năm")]},
    "稽": {"viet": "KÊ, KHỂ", "meaning_vi": "Xét. Như kê cổ [稽古] xét các sự tích xưa. Lời nói không có căn cứ gọi là vô kê chi ngôn [無稽之言].", "meo": "", "reading": "かんが.える とど.める", "vocab": [("稽古", "けいこ", "sự khổ luyện; sự luyện tập; sự rèn luyện; sự học tập"), ("滑稽", "こっけい", "buồn cười; ngố; ngố tàu; lố bịch; pha trò")]},
    "憾": {"viet": "HÁM", "meaning_vi": "Hối tiếc, ăn năn. Như di hám [遺憾] ân hận. Nguyễn Du [阮攸] : Bình sinh trực đạo vô di hám [平生直道無遺憾] (Âu Dương Văn Trung Công mộ [歐陽文忠公墓]) Bình sinh theo đường ngay, lòng không có gì hối tiếc.", "meo": "", "reading": "うら.む", "vocab": [("憾み", "うらみ", "lòng thương tiếc"), ("遺憾", "いかん", "đáng tiếc")]},
    "鍼": {"viet": "CHÂM", "meaning_vi": "cây kim, châm cứu", "meo": "Kim (金) có 10 (十) miệng (口) để châm cứu.", "reading": "はり", "vocab": [("鍼", "はり", "kim"), ("鍼灸", "しんきゅう", "châm cứu")]},
    "蒸": {"viet": "CHƯNG", "meaning_vi": "Lũ, bọn. Như chưng dân [蒸民].", "meo": "", "reading": "む.す む.れる む.らす", "vocab": [("蒸す", "むす", "chưng cách thủy"), ("蒸器", "むしき", "tàu chạy bằng hơi nước")]},
    "勾": {"viet": "CÂU", "meaning_vi": "Cong. Câu cổ [勾股] tên riêng của khoa học tính. Đo hình tam giác, đường ngang gọi là câu [勾], đường dọc gọi là cổ [股].", "meo": "", "reading": "かぎ ま.がる", "vocab": [("勾引", "こういん", "sự bắt giữ"), ("勾留", "こうりゅう", "sự giam cầm")]},
    "匂": {"viet": "(MÙI)", "meaning_vi": "fragrant, stink, glow, insinuate, (kokuji)", "meo": "", "reading": "にお.う にお.い にお.わせる", "vocab": [("匂い", "におい", "hơi hám"), ("匂う", "におう", "cảm thấy mùi; có mùi")]},
    "旬": {"viet": "TUẦN, QUÂN", "meaning_vi": "Tuần, mười ngày gọi là một tuần, một tháng có ba tuần. Từ mồng một đến mồng mười là thượng tuần [上旬], từ mười một đến hai mươi là trung tuần [中旬], từ hai mười mốt đến ba mươi là hạ tuần [下旬]. Nguyễn Du [阮攸] : Nhị tuần sở kiến đãn thanh san [二旬所見但青山] (Nam Quan đạo trung [南關道中]) Cả hai mươi ngày chỉ thấy núi xanh.", "meo": "", "reading": "ジュン シュン", "vocab": [("旬", "じゅん", "tuần; giai đoạn gồm 10 ngày"), ("一旬", "いちじゅん", "sự đập; tiếng đập")]},
    "殉": {"viet": "TUẪN", "meaning_vi": "Chết theo. Như quyết tâm nhất tử tuẫn phu [決心一死殉夫] quyết tâm chết theo chồng.", "meo": "", "reading": "ジュン", "vocab": [("殉国", "じゅんこく", "sự chết vì đất nước; sự hy sinh vì tổ quốc; hy sinh vì tổ quốc ."), ("殉教", "じゅんきょう", "sự chết vì nghĩa; sự chết vì đạo; sự chịu đoạ đày")]},
    "絢": {"viet": "HUYẾN", "meaning_vi": "Văn sức, trang sức sặc sỡ.", "meo": "", "reading": "ケン", "vocab": [("絢爛", "けんらん", "rực rỡ"), ("絢爛たる", "けんらんたる", "sáng chói")]},
    "刃": {"viet": "NHẬN", "meaning_vi": "Mũi nhọn.", "meo": "", "reading": "は やいば き.る", "vocab": [("刃", "は", "lưỡi (gươm); cạnh sắc"), ("両刃", "りょうば", "hai lưỡi")]},
    "忍": {"viet": "NHẪN", "meaning_vi": "Nhịn. Như làm việc khó khăn cũng cố làm cho được gọi là kiên nhẫn [堅忍], khoan dong cho người không vội trách gọi là dong nhẫn [容忍], v.v.", "meo": "", "reading": "しの.ぶ しの.ばせる", "vocab": [("忍び", "しのび", "/'spaiə/"), ("忍ぶ", "しのぶ", "chịu đựng; cam chịu")]},
    "梁": {"viet": "LƯƠNG", "meaning_vi": "Cái cầu. Chỗ ách yếu của sự vật gì gọi là tân lương [澤梁] nghĩa là như cái cầu mọi người đều phải nhờ đó mà qua vậy. Chỗ đắp bờ để dơm cá gọi là ngư lương [魚梁].", "meo": "", "reading": "はり うつばり うちばり やな はし", "vocab": [("梁", "やな", "xà"), ("梁木", "りょうぼく", "xà")]},
    "隠": {"viet": "ẨN", "meaning_vi": "Một dạng của chữ ẩn [隱].", "meo": "", "reading": "かく.す かく.し かく.れる かか.す よ.る", "vocab": [("隠す", "かくす", "bao bọc; che; che giấu; che đậy; giấu; giấu giếm"), ("隠る", "こもる", "da sống (chưa thuộc")]},
    "穏": {"viet": "ỔN", "meaning_vi": "Yên", "meo": "", "reading": "おだ.やか", "vocab": [("不穏", "ふおん", "tình trạng không yên ổn"), ("穏便", "おんびん", "khoan dung")]},
    "痩": {"viet": "SẤU", "meaning_vi": "gầy, ốm", "meo": "Bệnh (疒) ở (イ) nhà (冖) ai cũng cần (亻) lúa (禾) gạo (ク) để khỏe mạnh, nếu không sẽ gầy (痩) đi.", "reading": "や", "vocab": [("痩せる", "やせる", "gầy đi, giảm cân"), ("痩身", "そうしん", "thon thả, mảnh dẻ")]},
    "挿": {"viet": "SÁP", "meaning_vi": "Cho vào, thêm vào", "meo": "", "reading": "さ.す はさ.む", "vocab": [("挿す", "さす", "đính thêm; gắn vào; đeo dây lưng; cắm ."), ("挿入", "そうにゅう", "sự lồng vào; sự gài vào; sự sát nhập; sự hợp nhất")]},
    "亀": {"viet": "QUY, QUI", "meaning_vi": "Con rùa; hình dạng giống con rùa.", "meo": "", "reading": "かめ", "vocab": [("亀", "かめ", "rùa; con rùa"), ("海亀", "うみがめ", "đồi mồi .")]},
    "縄": {"viet": "THẰNG", "meaning_vi": "Sợi dây", "meo": "", "reading": "なわ ただ.す", "vocab": [("縄", "なわ", "dây thừng; dây chão"), ("縄墨", "なわすみ", "cờ hiệu")]},
    "陥": {"viet": "HÃM", "meaning_vi": "Vây hãm", "meo": "", "reading": "おちい.る おとしい.れる", "vocab": [("陥る", "おちいる", "rơi vào"), ("陥入", "おちいいり", "đổ")]},
    "稲": {"viet": "ĐẠO", "meaning_vi": "Cây lúa", "meo": "", "reading": "いね いな-", "vocab": [("稲", "いね", "lúa"), ("稲作", "いなさく", "trồng lúa")]},
    "葵": {"viet": "QUỲ", "meaning_vi": "Rau quỳ.", "meo": "", "reading": "あおい", "vocab": [("葵", "あおい", "cây thục quỳ"), ("戎葵", "えびすまもる", "cây thục quỳ")]},
    "廃": {"viet": "PHẾ", "meaning_vi": "Tàn phế, hoang phế.", "meo": "", "reading": "すた.れる すた.る", "vocab": [("廃り", "すたり", "bỏ hoang"), ("廃る", "すたる", "phế bỏ; vứt bỏ; lỗi thời; không lưu hành nữa")]},
    "庶": {"viet": "THỨ", "meaning_vi": "thứ dân, bình dân", "meo": "Mái nhà (广) che chở cho nhiều NGƯỜI (亻) để ai cũng có thể Ở (才).", "reading": "しょ", "vocab": [("庶民", "しょみん", "dân thường, thứ dân"), ("庶務", "しょむ", "tổng vụ, công việc hành chính")]},
    "遮": {"viet": "GIÀ", "meaning_vi": "Chận. Như già kích [遮擊] đánh chận hậu, đánh úp.", "meo": "", "reading": "さえぎ.る", "vocab": [("遮る", "さえぎる", "chắn"), ("遮光", "しゃこう", "bóng")]},
    "誠": {"viet": "THÀNH", "meaning_vi": "Thành thực, chân thực.", "meo": "", "reading": "まこと", "vocab": [("誠", "まこと", "niềm tin; sự tín nhiệm; sự trung thành; sự chân thành"), ("誠に", "まことに", "thực sự; thực tế; chân thực; rõ ràng")]},
    "斉": {"viet": "TỀ", "meaning_vi": "Nhất tề.", "meo": "", "reading": "そろ.う ひと.しい ひと.しく あたる はやい", "vocab": [("一斉", "いっせい", "cùng một lúc; đồng thanh; đồng loạt"), ("斉一", "せいいつ", "tính bằng")]},
    "弐": {"viet": "NHỊ", "meaning_vi": "hai", "meo": "Hai gạch ngang trên đầu tượng trưng cho 'hai'.", "reading": "に", "vocab": [("弐", "に", "hai (dùng trong văn bản trang trọng)")]},
    "賦": {"viet": "PHÚ", "meaning_vi": "Thu thuế, thu lấy những hoa lợi ruộng nương của dân để chi việc nước gọi là phú thuế [賦稅].", "meo": "", "reading": "フ ブ", "vocab": [("賦与", "ふよ", "sự phân bổ"), ("分賦", "ぶんふ", "sự giao việc")]},
    "戒": {"viet": "GIỚI", "meaning_vi": "Răn. Như khuyến giới [勸戒].", "meo": "", "reading": "いまし.める", "vocab": [("戒め", "いましめ", "lời cảnh báo"), ("十戒", "じっかい", "mười điều phật răn dạy .")]},
    "賊": {"viet": "TẶC", "meaning_vi": "Hại. Như tường tặc [戕賊] giết hại, kẻ làm hại dân gọi là dân tặc [民賊], kẻ làm hại nước gọi là quốc tặc [國賊].", "meo": "", "reading": "ゾク", "vocab": [("賊", "ぞく", "người nổi loạn"), ("兇賊", "きょうぞく", "côn đồ; kẻ hung ác")]},
    "拭": {"viet": "THỨC", "meaning_vi": "Lau. Như phất thức [拂拭] lau quét, thức lệ [拭淚] lau nước mắt.", "meo": "", "reading": "ぬぐ.う ふ.く", "vocab": [("拭う", "ぬぐう", "lau (mồ hôi)"), ("拭く", "ふく", "chùi")]},
    "或": {"viet": "HOẶC", "meaning_vi": "Hoặc, là lời nói còn ngờ, chưa quyết định hẳn. Như hoặc nhân [或人] hoặc người nào, hoặc viết [或曰] hoặc có kẻ nói rằng, v.v.", "meo": "", "reading": "あ.る あるい あるいは", "vocab": [("或", "ある", "mỗi; mỗi một; có một"), ("或は", "あるいは", "vàng")]},
    "惑": {"viet": "HOẶC", "meaning_vi": "Ngờ lạ. Như trí giả bất hoặc [智者不惑] kẻ khôn không có điều ngờ lạ.", "meo": "", "reading": "まど.う", "vocab": [("惑い", "まどい", "sự đánh lừa"), ("惑う", "まどう", "lúng túng; bối rối .")]},
    "栽": {"viet": "TÀI, TẢI", "meaning_vi": "Giồng (trồng).", "meo": "", "reading": "サイ", "vocab": [("栽", "さい", "trồng trọt ."), ("前栽", "せんざい", "vườn")]},
    "載": {"viet": "TÁI, TẠI, TẢI", "meaning_vi": "Chở. Nói về người thì gọi là thừa [乘], nói về xe thì gọi là tái [載]. Như tái dĩ hậu xa [載以後車] lấy xe sau chở về. Phàm dùng thuyền hay xe để chở đồ đều gọi là tái cả. Như mãn tái nhi quy [滿載而歸] xếp đầy thuyền chở về.", "meo": "", "reading": "の.せる の.る", "vocab": [("載る", "のる", "được đặt lên"), ("休載", "きゅうさい", "sự giảm nhẹ")]},
    "戴": {"viet": "ĐÁI", "meaning_vi": "đội, nhận, xin", "meo": "Chữ 士 (SĨ) đội lên đầu mũ 戈 (QUA) để nhận thưởng.", "reading": "いただく", "vocab": [("戴く", "いただく", "nhận, xin (khiêm nhường ngữ)"), ("戴冠式", "たいかんしき", "lễ đăng quang")]},
    "繊": {"viet": "TIÊM", "meaning_vi": "Thanh mảnh", "meo": "", "reading": "セン", "vocab": [("化繊", "かせん", "sợi tổng hợp; sự tổng hợp; sự kết hợp"), ("合繊", "ごうせん", "Sợi phíp tổng hợp")]},
    "戌": {"viet": "TUẤT", "meaning_vi": "Chi Tuất, chi thứ mười trong 12 chi. Từ 7 giờ tối đến 9 giờ tối là giờ Tuất.", "meo": "", "reading": "いぬ", "vocab": [("戌", "いぬ", "chó")]},
    "威": {"viet": "UY", "meaning_vi": "Oai, cái dáng tôn nghiêm đáng sợ gọi là uy. Như phát uy [發威] ra oai.", "meo": "", "reading": "おど.す おど.し おど.かす", "vocab": [("威令", "いれい", "uy quyền"), ("威信", "いしん", "thần thế")]},
    "滅": {"viet": "DIỆT", "meaning_vi": "Mất, tan mất.", "meo": "", "reading": "ほろ.びる ほろ.ぶ ほろ.ぼす", "vocab": [("不滅", "ふめつ", "bất diệt ."), ("滅亡", "めつぼう", "diệt vong")]},
    "蔑": {"viet": "MIỆT", "meaning_vi": "Không. Như miệt dĩ gia thử [蔑以加此] không gì hơn thế nữa.", "meo": "", "reading": "ないがしろ なみ.する くらい さげす.む", "vocab": [("蔑む", "さげすむ", "coi thường; khinh miệt"), ("蔑ろ", "ないがしろ", "Việc coi thường; sự khinh miệt")]},
    "茂": {"viet": "MẬU", "meaning_vi": "Tốt, cây cỏ tốt tươi. Như trúc bao tùng mậu [竹苞松茂] tùng trúc tốt tươi.", "meo": "", "reading": "しげ.る", "vocab": [("茂み", "しげみ", "bụi cây ."), ("茂る", "しげる", "rậm rạp; um tùm; xanh tốt")]},
    "虚": {"viet": "HƯ, KHƯ", "meaning_vi": "Cũng như chữ hư [虛].", "meo": "", "reading": "むな.しい うつ.ろ", "vocab": [("虚ろ", "うつろ", "để trống"), ("虚仮", "こけ", "sự điên rồ; hành động đại dột")]},
    "嘘": {"viet": "HƯ", "meaning_vi": "Dị dạng của chữ [噓].", "meo": "", "reading": "うそ ふ.く", "vocab": [("嘘", "うそ", "bịa chuyện"), ("嘘つき", "うそつき", "kẻ nói dối; kẻ nói láo; loại bốc phét; loại ba hoa; kẻ nói phét")]},
    "戯": {"viet": "HÍ", "meaning_vi": "Giản thể của chữ 戱", "meo": "", "reading": "たわむ.れる ざ.れる じゃ.れる", "vocab": [("戯れ", "たわむれ", "trò chơi; trò đùa; thể thao; giải trí ."), ("戯作", "げさく", "điều hư cấu")]},
    "膚": {"viet": "PHU", "meaning_vi": "Da ngoài. Sự tai hại đến thân gọi là thiết phu chi thống [切膚之痛] đau như cắt da.", "meo": "", "reading": "はだ", "vocab": [("膚", "はだ", "da"), ("人膚", "ひとはだ", "Da; sức nóng thân thể .")]},
    "慮": {"viet": "LỰ, LƯ", "meaning_vi": "Nghĩ toan. Nghĩ định toan làm một sự gì gọi là lự.", "meo": "", "reading": "おもんぱく.る おもんぱか.る", "vocab": [("慮り", "おもんばかり", "sự suy nghĩ"), ("慮る", "おもんばかる", "cân nhắc")]},
    "虎": {"viet": "HỔ", "meaning_vi": "Con hổ.", "meo": "", "reading": "とら", "vocab": [("虎", "とら", "hổ"), ("両虎", "りょうこ", "người anh hùng")]},
    "虜": {"viet": "LỖ", "meaning_vi": "Tù binh. Bắt sống được quân địch gọi là lỗ [虜], chém đầu được quân giặc gọi là hoạch [獲].", "meo": "", "reading": "とりこ とりく", "vocab": [("虜", "とりこ", "bị bắt giữ"), ("俘虜", "ふりょ", "bị bắt giữ")]},
    "虐": {"viet": "NGƯỢC", "meaning_vi": "Ác, tai ngược, nghiệt. Như ngược đãi [虐待] đối xử nghiệt ác, ngược chánh [虐政] chánh trị tàn ác.", "meo": "", "reading": "しいた.げる", "vocab": [("虐め", "いじめ", "sự chòng ghẹo"), ("虐待", "ぎゃくたい", "đọa đầy")]},
    "糾": {"viet": "CỦ, KIỂU", "meaning_vi": "Dây chặp ba lần, vì thế cái gì do mọi cái kết hợp lại mà thành đều gọi là củ. Như củ chúng [糾眾] nhóm họp mọi người.", "meo": "", "reading": "ただ.す", "vocab": [("糾合", "きゅうごう", "sự tập hợp; sự tập trung; tập hợp; tập trung ."), ("糾問", "きゅうもん", "sự thẩm vấn; thẩm vấn; sự tra hỏi; tra hỏi")]},
    "猶": {"viet": "DO, DỨU", "meaning_vi": "Con do, giống như con khỉ, tính hay ngờ, nghe tiếng người leo ngay lên cây, không thấy người mới lại xuống. Vì thế mới gọi những người hay ngờ, không quả quyết là do dự [猶豫].", "meo": "", "reading": "なお", "vocab": [("猶予", "ゆうよ", "sự trì hoãn; sự để chậm lại; sự hoãn lại"), ("猶予なく", "ゆうよなく", "sự nhanh chóng .")]},
    "遵": {"viet": "TUÂN", "meaning_vi": "Lần theo.", "meo": "", "reading": "ジュン", "vocab": [("遵奉", "じゅんぽう", "sự tuân thủ; sự tuân theo; tuân thủ ."), ("遵守", "じゅんしゅ", "sự tuân thủ; sự bảo đảm .")]},
    "樽": {"viet": "TÔN", "meaning_vi": "Cũng như chữ tôn [尊] nghĩa là cái chén.", "meo": "", "reading": "たる", "vocab": [("樽", "たる", "thùng"), ("酒樽", "さかだる", "thùng rượu .")]},
    "噂": {"viet": "ĐỒN", "meaning_vi": "Tin đồn", "meo": "", "reading": "うわさ", "vocab": [("噂", "うわさ", "lời đồn đại; tin đồn; tiếng đồn"), ("噂する", "うわさする", "đồn; nói chuyện phiếm; bàn tán; buôn chuyện")]},
    "鱒": {"viet": "TỖN", "meaning_vi": "Cá tỗn, cá chầy, cá rói.", "meo": "", "reading": "ます", "vocab": [("鱒", "ます", "cá hồi ."), ("姫鱒", "ひめます", "Cá hồi đỏ .")]},
    "旦": {"viet": "ĐÁN", "meaning_vi": "Sớm, lúc trời mới sáng gọi là đán.", "meo": "", "reading": "あき.らか あきら ただし あさ あした", "vocab": [("一旦", "いったん", "một khi"), ("元旦", "がんたん", "ngày mùng một Tết; sáng mùng một Tết")]},
    "但": {"viet": "ĐÃN", "meaning_vi": "Những. Lời nói chuyển câu. Lý Thương Ẩn [李商隱] : Hiểu kính đãn sầu vân mấn cải [曉鏡但愁雲鬢改] (Vô đề [無題]) Sớm mai soi gương, những buồn cho tóc mây đã đổi.", "meo": "", "reading": "ただ.し", "vocab": [("但し", "ただし", "tuy nhiên; nhưng"), ("但書", "ただしがき", "/prə'vaizouz/")]},
    "胆": {"viet": "ĐẢM", "meaning_vi": "Tục dùng như chữ đảm [膽].", "meo": "", "reading": "きも", "vocab": [("胆", "きも", "mật ."), ("剛胆", "ごうたん", "tính dũng cảm")]},
    "壇": {"viet": "ĐÀN", "meaning_vi": "Cái đàn.", "meo": "", "reading": "ダン タン", "vocab": [("壇", "だん", "bục"), ("壇上", "だんじょう", "bàn thờ .")]},
    "宣": {"viet": "TUYÊN", "meaning_vi": "To lớn. Như tuyên thất [宣室] cái nhà to, vì thế nên tường vách xây tới sáu tấc cũng gọi là tuyên, thông dùng như chữ [瑄].", "meo": "", "reading": "のたま.う", "vocab": [("不宣", "ふせん", "Bạn chân thành! ."), ("宣伝", "せんでん", "sự tuyên truyền; thông tin tuyên truyền; sự công khai .")]},
    "喧": {"viet": "HUYÊN", "meaning_vi": "Dức lác. Đặng Trần Côn [鄧陳琨] : Liệp liệp tinh kỳ hề xuất tái sầu, Huyên huyên tiêu cổ hề từ gia oán [獵獵旌旗兮出塞 愁，喧喧簫鼓兮辭家怨] (Chinh Phụ ngâm [征婦吟]) Cờ tinh cờ kỳ bay rộn ràng, giục lòng sầu ra ải, Tiếng sáo tiếng trống inh ỏi, lẫn tiếng oán lìa nhà. Đoàn Thị Điểm dịch thơ : Bóng cờ tiếng trống xa xa, Sầu lên ngọn ải, oán ra cửa phòng.", "meo": "", "reading": "やかま.しい かまびす.しい", "vocab": [("喧嘩", "けんか", "sự cà khịa; sự cãi cọ; sự tranh chấp; cà khịa; cãi cọ; tranh chấp"), ("喧噪", "けんそう", "ồn ào")]},
    "恒": {"viet": "HẰNG, CẮNG, CĂNG", "meaning_vi": "Tục dùng như chữ hằng [恆].", "meo": "", "reading": "つね つねに", "vocab": [("恒久", "こうきゅう", "sự vĩnh cửu; cái không thay đổi; sự vĩnh viễn"), ("恒例", "こうれい", "thói quen; thông lệ; thường lệ")]},
    "垣": {"viet": "VIÊN", "meaning_vi": "Tường thấp.", "meo": "", "reading": "かき", "vocab": [("垣", "かき", "hàng rào"), ("中垣", "なかがき", "Hàng rào ở giữa .")]},
    "褻": {"viet": "TIẾT", "meaning_vi": "Áo lót mình.", "meo": "", "reading": "けが.れる な.れる", "vocab": [("褻", "け", "bẩn thỉu"), ("猥褻", "わいせつ", "sự tục tĩu")]},
    "睦": {"viet": "MỤC", "meaning_vi": "Hòa kính, tin, thân.", "meo": "", "reading": "むつ.まじい むつ.む むつ.ぶ", "vocab": [("和睦", "わぼく", "sự hoà giải"), ("敦睦", "あつしあつし", "thương yêu")]},
    "陵": {"viet": "LĂNG", "meaning_vi": "Đống đất to, cái gò.", "meo": "", "reading": "みささぎ", "vocab": [("陵", "みささぎ", "lăng mộ hoàng đế ."), ("丘陵", "きゅうりょう", "đồi núi")]},
    "凌": {"viet": "LĂNG", "meaning_vi": "Lớp váng, nước giá tích lại từng lớp nọ lớp kia gọi là lăng.", "meo": "", "reading": "しの.ぐ", "vocab": [("凌ぐ", "しのぐ", "át hẳn; áp đảo; vượt trội"), ("凌ぎ場", "しのぎじょう", "chỗ che")]},
    "喚": {"viet": "HOÁN", "meaning_vi": "Kêu, gọi. Nguyễn Trãi [阮廌] : Hoán hồi ngọ mộng chẩm biên cầm [喚回午夢枕邊禽] (Đề Trình xử sĩ Vân oa đồ [題程處士雲窩圖]) Gọi tỉnh giấc mộng trưa, sẵn tiếng chim bên gối.", "meo": "", "reading": "わめ.く", "vocab": [("喚く", "わめく", "kêu lên; gào thét ."), ("叫喚", "きょうかん", "tiếng kêu; sự la hét")]},
    "姫": {"viet": "CƠ", "meaning_vi": "Công chúa", "meo": "", "reading": "ひめ ひめ-", "vocab": [("姫", "ひめ", "cô gái quí tộc; tiểu thư"), ("姫君", "ひめぎみ", "công chúa .")]},
    "監": {"viet": "GIAM, GIÁM", "meaning_vi": "Soi xét, coi sóc. Như giam đốc [監督] người coi sóc công việc của kẻ dưới.", "meo": "", "reading": "カン", "vocab": [("監事", "かんじ", "người quản lý"), ("監修", "かんしゅう", "sự trông nom")]},
    "鑑": {"viet": "GIÁM", "meaning_vi": "Cái gương soi. Ngày xưa dùng đồng làm gương soi gọi là giám. Đem các việc hỏng trước chép vào sách để làm gương soi cũng gọi là giám. Như ông Tư Mã Quang [司馬光] làm bộ Tư trị thông giám [資治通鑑] nghĩa là pho sử để soi vào đấy mà giúp thêm các cách trị dân.", "meo": "", "reading": "かんが.みる かがみ", "vocab": [("亀鑑", "きかん", "kiểu mẫu"), ("鑑別", "かんべつ", "phân biệt")]},
    "艦": {"viet": "HẠM", "meaning_vi": "Tàu trận. Nay gọi quân đánh trên mặt bể là hạm đội [艦隊], tàu trận gọi là quân hạm [軍艦].", "meo": "", "reading": "カン", "vocab": [("艦", "かん", "hạm; trạm"), ("乗艦", "じょうかん", "sự quy định")]},
    "藍": {"viet": "LAM", "meaning_vi": "Cây chàm.", "meo": "", "reading": "あい", "vocab": [("藍", "あい", "màu chàm"), ("伽藍", "がらん", "đến")]},
    "濫": {"viet": "LẠM, LÃM, CÃM", "meaning_vi": "Giàn giụa.", "meo": "", "reading": "みだ.りに みだ.りがましい", "vocab": [("濫伐", "らんばつ", "sự chặt phá bừa bài ."), ("濫作", "らんさく", "sự sản xuất quá độ .")]},
    "徳": {"viet": "ĐỨC", "meaning_vi": "Một cách viết của chữ đức [德]. Trần Nhân Tông [陳仁宗] : Nhất thị đồng nhân thiên tử đức [一視同仁天子徳] (Họa Kiều Nguyên Lãng vận [和喬元朗韻]) Khắp thấy \"đồng nhân\" (cùng thương người) là đức của bậc thiên tử.", "meo": "", "reading": "トク", "vocab": [("お徳", "おとく", "sự tiết kiệm; có tính kinh tế"), ("一徳", "かずのり", "đức")]},
    "聴": {"viet": "THÍNH", "meaning_vi": "Cũng như chữ thính [聽].", "meo": "", "reading": "き.く ゆる.す", "vocab": [("聴く", "きく", "nghe; lắng nghe"), ("傍聴", "ぼうちょう", "sự nghe; việc nghe")]},
    "憲": {"viet": "HIẾN", "meaning_vi": "Pháp, yết các điều pháp luật lên cho người biết mà theo gọi là hiến. Nước nào lấy pháp luật mà trị nước gọi là lập hiến quốc [立憲國]. Nguyễn Trãi [阮廌] : Đáo để chung đầu hiến võng trung [到底終投憲網中] (Hạ tiệp [賀捷]) Cuối cùng rồi phải sa vào lưới pháp luật.", "meo": "", "reading": "ケン", "vocab": [("憲兵", "けんぺい", "hiến binh"), ("合憲", "ごうけん", "sức mạnh")]},
    "懐": {"viet": "HOÀI", "meaning_vi": "Hoài cổ, nhớ nhung", "meo": "", "reading": "ふところ なつ.かしい なつ.かしむ なつ.く なつ.ける なず.ける いだ.く おも.う", "vocab": [("懐", "ふところ", "ngực áo; ngực; bộ ngực"), ("懐く", "なつく", "theo")]},
    "之": {"viet": "CHI", "meaning_vi": "Chưng, dùng về lời nói liền nối nhau. Như đại học chi đạo [大學之道] chưng đạo đại học.", "meo": "", "reading": "の これ おいて ゆく この", "vocab": [("之", "これ", "Đây; này ."), ("鳥之巣", "とりのす", "tổ chim .")]},
    "乏": {"viet": "PHẠP", "meaning_vi": "Thiếu, không có đủ.", "meo": "", "reading": "とぼ.しい とも.しい", "vocab": [("乏しい", "とぼしい", "cùng khốn; bần cùng"), ("欠乏", "けつぼう", "điêu đứng")]},
    "芝": {"viet": "CHI", "meaning_vi": "Một loài cỏ như nấm, mọc ở các cây đã chết, hình như cái nấm, cứng nhẵn nhụi, có sáu sắc xanh, đỏ, vàng, trắng, đen, tía. Người xưa cho đó là cỏ báo điềm tốt lành, nên gọi là linh chi [靈芝].", "meo": "", "reading": "しば", "vocab": [("芝", "しば", "cỏ; cỏ thấp sát đất ."), ("芝居", "しばい", "kịch")]},
    "慢": {"viet": "MẠN", "meaning_vi": "kiêu căng, chậm trễ", "meo": "Trái tim (忄) lúc nào cũng CHẬM (曼) chạp, từ tốn vì luôn nghĩ mình là số 1, kiêu MẠN.", "reading": "まん", "vocab": [("不満", "ふまん", "bất mãn"), ("自慢", "じまん", "tự mãn, khoe khoang")]},
    "漫": {"viet": "MẠN", "meaning_vi": "tản mạn, tùy tiện", "meo": "Nước (氵) tràn ra hai bên (罒) vì quá mười (蔓) năm.", "reading": "まん", "vocab": [("漫画", "まんが", "truyện tranh"), ("漫才", "まんざい", "hài kịch, tấu hài")]},
    "蔓": {"viet": "MẠN", "meaning_vi": "Các loài thực vật rò bò dài dưới đất gọi là mạn.", "meo": "", "reading": "はびこ.る つる", "vocab": [("手蔓", "てづる", "ảnh hưởng"), ("蔓草", "つるくさ", "cây leo .")]},
    "鰻": {"viet": "MAN", "meaning_vi": "Cá sộp, cá chình. Tục gọi là man li [鰻鱺]. Cũng gọi là bạch thiện [白鱔].", "meo": "", "reading": "うなぎ", "vocab": [("鰻", "うなぎ", "con lươn"), ("鰻登り", "うなぎのぼり", "Sự thúc đẩy nhanh; tăng nhanh vùn vụt")]},
    "称": {"viet": "XƯNG, XỨNG", "meaning_vi": "Giản thể của chữ [稱].", "meo": "", "reading": "たた.える とな.える あ.げる かな.う はか.り はか.る ほめ.る", "vocab": [("称", "しょう", "tên; nhãn hiệu ."), ("人称", "にんしょう", "nhân xưng")]},
    "弥": {"viet": "DI", "meaning_vi": "Tục dùng như chữ di [彌].", "meo": "", "reading": "いや や あまねし いよいよ とおい ひさし ひさ.しい わた.る", "vocab": [("弥四", "わたるよん", "ông bầu"), ("弥次", "わたるじ", "sự chế giễu")]},
    "貴": {"viet": "QUÝ", "meaning_vi": "Sang, quý hiển. Như công danh phú quý [功名富貴] có công nghiệp, có tiếng tăm, được giàu sang. Dòng họ cao sang gọi là quý tộc [貴族].", "meo": "", "reading": "たっと.い とうと.い たっと.ぶ とうと.ぶ", "vocab": [("貴い", "とうとい", "quý giá; quý báu; tôn quý; cao quý ."), ("貴い", "たっとい", "quý giá; quý báu; tôn quý; cao quý")]},
    "遺": {"viet": "DI, DỊ", "meaning_vi": "Bỏ sót, mất, vô ý bỏ mất đi gọi là di. Như thập di [拾遺] nhặt nhạnh các cái bỏ sót, bổ di [補遺] bù các cái bỏ sót.", "meo": "", "reading": "のこ.す", "vocab": [("遺す", "のこす", "để lại"), ("遺伝", "いでん", "di truyền")]},
    "潰": {"viet": "HỘI", "meaning_vi": "Vỡ ngang. Nước phá ngang bờ chắn mà chảy tóe vào gọi là hội. Như hội đê [潰隄] vỡ đê.", "meo": "", "reading": "つぶ.す つぶ.れる つい.える", "vocab": [("潰す", "つぶす", "giết (thời gian)"), ("潰乱", "かいらん", "sự hối lộ")]},
    "憤": {"viet": "PHẪN, PHẤN", "meaning_vi": "Tức giận, uất ức quá gọi là phẫn [憤]. Như phẫn nộ [憤怒] giận dữ.", "meo": "", "reading": "いきどお.る", "vocab": [("憤り", "いきどおり", "sự phẫn uất"), ("憤る", "いきどおる", "phẫn uất")]},
    "墳": {"viet": "PHẦN, PHẪN, BỔN", "meaning_vi": "Cái mả cao.", "meo": "", "reading": "フン", "vocab": [("墳丘", "ふんきゅう", "nấm mồ"), ("古墳", "こふん", "mộ cổ")]},
    "脹": {"viet": "TRƯỚNG", "meaning_vi": "Trương. Bụng đầy rán lên gọi là phúc trướng [腹脹]. Nề sưng gọi là thũng trướng [腫脹].", "meo": "", "reading": "は.れる ふく.らむ ふく.れる", "vocab": [("脹れ", "ふくれ", "nhọt"), ("脹らみ", "ふくらみ", "chỗ phình")]},
    "髭": {"viet": "TÌ", "meaning_vi": "Râu trên mồm. Râu mọc chung quanh trên mồm trên gọi là tì.", "meo": "", "reading": "ひげ くちひげ", "vocab": [("髭", "ひげ", "râu ."), ("口髭", "くちひげ", "râu mép")]},
    "脊": {"viet": "TÍCH", "meaning_vi": "Xương sống, có 24 đốt.", "meo": "", "reading": "せ せい", "vocab": [("脊柱", "せきちゅう", "cột sống; xương sống lưng"), ("脊梁", "せきりょう", "xương sống")]},
    "滑": {"viet": "HOẠT, CỐT", "meaning_vi": "Trơn, nhẵn.", "meo": "", "reading": "すべ.る なめ.らか", "vocab": [("滑", "なめら", "lỗ hổng"), ("滑り", "すべり", "sự trượt")]},
    "享": {"viet": "HƯỞNG", "meaning_vi": "Dâng, đem đồ lễ lên dâng người trên hay đem cúng tế gọi là hưởng.", "meo": "", "reading": "う.ける", "vocab": [("享受", "きょうじゅ", "sự hưởng thụ; hưởng thụ; nhận; hưởng"), ("享有", "きょうゆう", "sự được hưởng; hưởng; được hưởng")]},
    "郭": {"viet": "QUÁCH", "meaning_vi": "Cái thành ngoài.", "meo": "", "reading": "くるわ", "vocab": [("一郭", "いっかく", "sự rào lại"), ("郭公", "かっこう", "chim cúc cu; tiếng chim cúc cu .")]},
    "熟": {"viet": "THỤC", "meaning_vi": "Chín.", "meo": "", "reading": "う.れる", "vocab": [("熟", "つくづく", "tỉ mỉ; sâu sắc; thật sự ."), ("熟す", "こなす", "đập vỡ; nghiền vụn; tiêu hoá; nắm vững; chín; chín muồi .")]},
    "醇": {"viet": "THUẦN", "meaning_vi": "Rượu ngon, rượu nặng.", "meo": "", "reading": "もっぱら こい あつい", "vocab": [("醇朴", "じゅんぼく", "tính chất giản dị"), ("醇正", "あつしただし", "trong")]},
    "倉": {"viet": "THƯƠNG, THẢNG", "meaning_vi": "Cái bịch đựng thóc.", "meo": "", "reading": "くら", "vocab": [("倉", "くら", "nhà kho; kho"), ("営倉", "えいそう", "phòng nghỉ của lính gác")]},
    "創": {"viet": "SANG, SÁNG", "meaning_vi": "Bị thương đau. Như trọng sang [重創] bị thương nặng.", "meo": "", "reading": "つく.る はじ.める きず けず.しける", "vocab": [("創", "そう", "bắt đầu; khởi nguồn"), ("創作", "そうさく", "tác phẩm .")]},
    "槍": {"viet": "THƯƠNG, SANH", "meaning_vi": "Đẽo gỗ làm đồ binh.", "meo": "", "reading": "やり", "vocab": [("槍", "やり", "cái giáo"), ("槍先", "やりさき", "mũi giáo")]},
    "蒼": {"viet": "THƯƠNG, THƯỞNG", "meaning_vi": "Sắc cỏ xanh. Phàm cái gì xanh sẫm đều gọi là thương. Như thương giang [蒼江] sông biếc, thương hải [蒼海] bể xanh, thương thương [蒼蒼] trời xanh, v.v.", "meo": "", "reading": "あお.い", "vocab": [("蒼い", "あおい", "xanh"), ("蒼然", "そうぜん", "xanh")]},
    "没": {"viet": "MỘT", "meaning_vi": "Giản thể của chữ [沒].", "meo": "", "reading": "おぼ.れる しず.む ない", "vocab": [("没", "ぼつ", "cái chết; sự chấm hết"), ("没入", "ぼつにゅう", "sự chìm; sự bị nhấn chìm")]},
    "股": {"viet": "CỔ", "meaning_vi": "Đùi vế.", "meo": "", "reading": "また もも", "vocab": [("股", "また", "chỗ giao nhau; chỗ chia tách ra làm hai"), ("股", "もも", "đùi .")]},
    "疫": {"viet": "DỊCH", "meaning_vi": "Bệnh ôn dịch, bệnh nào có thể lây ra mọi người được gọi là dịch.", "meo": "", "reading": "エキ ヤク", "vocab": [("免疫", "めんえき", "sự miễn dịch ."), ("疫学", "えきがく", "khoa nghiên cứu bệnh dịch")]},
    "搬": {"viet": "BÀN, BAN", "meaning_vi": "Trừ hết, dọn sạch.", "meo": "", "reading": "ハン", "vocab": [("伝搬", "でんぱん", "sự lan truyền; sự truyền lan"), ("搬入", "はんにゅう", "chở vào")]},
    "盤": {"viet": "BÀN", "meaning_vi": "Cái mâm.", "meo": "", "reading": "バン", "vocab": [("盤", "ばん", "đĩa; khay ."), ("円盤", "えんばん", "đĩa")]},
    "殻": {"viet": "XÁC", "meaning_vi": "Tục dùng như chữ xác [殼].", "meo": "", "reading": "から がら", "vocab": [("殻", "から", "vỏ (động thực vật); trấu (gạo); lớp bao ngoài; vỏ ngoài"), ("介殻", "かいから", "sự cải cách")]},
    "穀": {"viet": "CỐC, LỘC, DỤC", "meaning_vi": "Lúa, loài thực vật dùng để ăn. Như lúa tẻ lúa nếp đều gọi là cốc. Ngũ cốc [五穀] năm thứ cốc : đạo, thử, tắc, mạch, thục [稻黍稷麥菽] lúa gié, lúa nếp, lúa tắc, lúa tẻ, đậu.", "meo": "", "reading": "コク", "vocab": [("五穀", "ごこく", "ngũ cốc"), ("穀倉", "こくそう", "kho ngũ cốc .")]},
    "毅": {"viet": "NGHỊ", "meaning_vi": "Quả quyết, cứng cỏi, quyết chí không ai lay động được gọi là nghị lực [毅力].", "meo": "", "reading": "つよ.い", "vocab": [("剛毅", "ごうき", "sự chịu đựng ngoan cường; sự dũng cảm chịu đựng"), ("毅然", "きぜん", "sự chịu đựng ngoan cường; sự dũng cảm chịu đựng")]},
    "鍛": {"viet": "ĐOÁN", "meaning_vi": "Gió sắt, rèn sắt.", "meo": "", "reading": "きた.える", "vocab": [("鍛冶", "たんや", "thợ rèn"), ("鍛工", "たんこう", "thợ rèn")]},
    "刹": {"viet": "SÁT", "meaning_vi": "Giết, hãm hại, ngôi chùa", "meo": "Con dao (刂) chặt cây (木), sát sinh!", "reading": "さつ", "vocab": [("殺到", "さっとう", "Ùn ùn kéo đến, đổ xô"), ("刹那", "せつな", "Khoảnh khắc, chốc lát")]},
    "壱": {"viet": "NHẤT", "meaning_vi": "số một, một (thay cho 一)", "meo": "", "reading": "ひとつ", "vocab": [("壱", "いち", "một"), ("金壱万円", "きんいちまんえん", "một vạn yên")]},
    "庇": {"viet": "TÍ", "meaning_vi": "Che chở. Như tí hộ [庇護] che chở giúp giữ cho.", "meo": "", "reading": "ひさし おお.う かば.う", "vocab": [("庇う", "かばう", "bao che; che giấu"), ("目庇", "まびさし", "lưới trai mũ")]},
    "陛": {"viet": "BỆ", "meaning_vi": "Thềm nhà vua.", "meo": "", "reading": "ヘイ", "vocab": [("陛下", "へいか", "bệ hạ"), ("両陛下", "りょうへいか", "vua và hoàng hậu; hoàng đế và hoàng hậu .")]},
    "蛇": {"viet": "XÀ, DI", "meaning_vi": "Con rắn. Một năm lột xác một lần gọi là xà thoái [蛇退].", "meo": "", "reading": "へび", "vocab": [("蛇", "へび", "rắn"), ("蛇体", "じゃたい", "rắn; hình rắn")]},
    "舵": {"viet": "ĐÀ", "meaning_vi": "Cũng như chữ đà [柁], cái bánh lái thuyền.", "meo": "", "reading": "かじ", "vocab": [("舵", "かじ", "bánh lái"), ("舵手", "だしゅ", "người lái tàu thuỷ")]},
    "陀": {"viet": "ĐÀ", "meaning_vi": "Chỗ đất gập ghềnh. $ Có khi viết là [阤].", "meo": "", "reading": "けわ.しい ななめ", "vocab": [("仏陀", "ぶっだ", "phật"), ("仏陀", "ぶつだ", "Phật thích ca; Phật Đà .")]},
    "而": {"viet": "NHI", "meaning_vi": "và, rồi thì, nhưng", "meo": "Như Răng (mà điền vào chỗ trống hàm răng)", "reading": "して", "vocab": [("而して", "そして", "và, rồi thì"), ("然し而も", "しかしながら", "tuy nhiên, nhưng mà")]},
    "儒": {"viet": "NHO", "meaning_vi": "Học trò. Tên chung của những người có học. Như thạc học thông nho [碩學通儒] người học giỏi hơn người.", "meo": "", "reading": "ジュ", "vocab": [("儒", "じゅ", "đạo Khổng; người theo đạo Khổng"), ("侏儒", "しゅじゅ", "lùn")]},
    "濡": {"viet": "NHU, NHI", "meaning_vi": "Sông Nhu.", "meo": "", "reading": "ぬれ.る ぬら.す ぬ.れる ぬ.らす うるお.い うるお.う うるお.す", "vocab": [("濡らす", "ぬらす", "dấn"), ("濡れる", "ぬれる", "đằm")]},
    "耐": {"viet": "NẠI", "meaning_vi": "Chịu nhịn. Như nại cơ [耐飢] chịu nhịn được đói, nại khổ [耐苦] chịu nhịn được khổ. Nguyễn Du [阮攸] : Nại đắc phong sương toàn nhĩ tính [耐得風霜全爾性] (Thành hạ khí mã [城下棄馬]) Chịu được phong sương, trọn tánh trời.", "meo": "", "reading": "た.える", "vocab": [("耐久", "たいきゅう", "sự chịu đựng"), ("耐乏", "たいぼう", "sự nghiêm khắc")]},
    "瑞": {"viet": "THỤY", "meaning_vi": "Tên chung của ngọc khuê ngọc bích, đời xưa dùng làm dấu hiệu cho nên gọi là thụy.", "meo": "", "reading": "みず- しるし", "vocab": [("瑞々しい", "みずみずしい", "trẻ trung và sôi nổi .")]},
    "喘": {"viet": "SUYỄN", "meaning_vi": "Thở gằn, khí bực tức thở mau quá độ. Ta gọi là bệnh suyễn. Như khí suyễn nan đương [氣喘難當] ngộp thở khó chịu.", "meo": "", "reading": "あえ.ぐ せき", "vocab": [("喘ぎ", "あえぎ", "bệnh hen"), ("喘ぐ", "あえぐ", "sự thở hổn hển")]},
    "頃": {"viet": "KHOẢNH, KHUYNH, KHUỂ", "meaning_vi": "Thửa ruộng trăm mẫu.", "meo": "", "reading": "ころ ごろ しばら.く", "vocab": [("頃", "ころ", "dạo"), ("頃", "ごろ", "vào khoảng .")]},
    "煩": {"viet": "PHIỀN", "meaning_vi": "Phiền (không được giản dị).", "meo": "", "reading": "わずら.う わずら.わす うるさ.がる うるさ.い", "vocab": [("煩", "はん", "điều lo lắng"), ("煩い", "うるさい", "chán ghét; đáng ghét")]},
    "須": {"viet": "TU", "meaning_vi": "Đợi. Như tương tu thậm ân [相須甚殷] cùng đợi rất gấp.", "meo": "", "reading": "すべから.く すべし ひげ まつ もち.いる もと.める", "vocab": [("必須", "ひっす", "cần thiết"), ("須臾", "しゅゆ", "lúc")]},
    "謡": {"viet": "DAO", "meaning_vi": "Câu vè, bài hát có chương có khúc gọi là ca [歌], không có chương có khúc gọi là dao [謡]. Như phong dao, ca dao, v.v.", "meo": "", "reading": "うた.い うた.う", "vocab": [("謡", "うたい", "sự kể lại"), ("謡う", "うたう", "hát")]},
    "揺": {"viet": "DAO", "meaning_vi": "Dao động.", "meo": "", "reading": "ゆ.れる ゆ.る ゆ.らぐ ゆ.るぐ ゆ.する ゆ.さぶる ゆ.すぶる うご.く", "vocab": [("揺", "ゆら", "sự rung động; sự làm rung động"), ("揺り", "ゆり", "sự rung động; sự làm rung động")]},
    "遥": {"viet": "DIÊU, DAO", "meaning_vi": "Giản thể của chữ [遙].", "meo": "", "reading": "はる.か", "vocab": [("遥か", "はるか", "xa xưa; xa; xa xôi"), ("遥かに", "はるかに", "xa")]},
    "陶": {"viet": "ĐÀO, DAO", "meaning_vi": "Đồ sành. Đồ gốm.", "meo": "", "reading": "トウ", "vocab": [("陶冶", "とうや", "sự dạy dỗ"), ("陶器", "とうき", "đồ gốm")]},
    "淘": {"viet": "ĐÀO", "meaning_vi": "Vo gạo.", "meo": "", "reading": "よな.げる", "vocab": [("淘汰", "とうた", "Sự chọn lọc (tự nhiên)"), ("雌雄淘汰", "しゆうとうた", "sự chọn lọc giới tính")]},
    "迭": {"viet": "ĐIỆT", "meaning_vi": "Đắp đổi, thay đổi phiên, lần lượt. Nguyễn Trãi [阮廌] : Hoan bi ưu lạc điệt vãng lai [歡悲憂樂迭往來] (Côn sơn ca [崑山歌]) Vui buồn lo sướng đổi thay nhau.", "meo": "", "reading": "テツ", "vocab": [("更迭", "こうてつ", "di dịch"), ("更迭する", "こうてつする", "đắc cách .")]},
    "侯": {"viet": "HẦU", "meaning_vi": "Tước hầu. Các nhà đế vương đặt ra năm tước để phong cho bầy tôi, tước hầu là tước thứ hai trong năm tước : công, hầu, bá, tử, nam [公侯伯子男]. Đời phong kiến, thiên tử phong họ hàng công thần ra làm vua các xứ, gọi là vua chư hầu, đời sau nhân thế, mượn làm tiếng gọi các quan sang. Như quân hầu [君侯], ấp hầu [邑侯], v.v.", "meo": "", "reading": "コウ", "vocab": [("侯", "こう", "hầu"), ("公侯", "こうこう", "công hầu .")]},
    "喉": {"viet": "HẦU", "meaning_vi": "Cổ họng hơi. Như yết hầu [咽喉] cổ họng.", "meo": "", "reading": "のど", "vocab": [("喉", "のど", "cuống họng"), ("喉仏", "のどぼとけ", "Cục yết hầu .")]},
    "蟹": {"viet": "GIẢI", "meaning_vi": "Con cua.", "meo": "", "reading": "かに", "vocab": [("蟹", "かに", "con cua; cua"), ("蟹座", "かにざ", "bệnh ung thư")]},
    "麗": {"viet": "LỆ, LI", "meaning_vi": "Đẹp. Như diễm lệ [豔麗] tươi đẹp, đẹp lộng lẫy.", "meo": "", "reading": "うるわ.しい うら.らか", "vocab": [("麗人", "れいじん", "người phụ nữ đẹp; người diễm lệ; người yêu kiều; mỹ nhân"), ("佳麗", "かれい", "vẻ đẹp")]},
    "麓": {"viet": "LỘC", "meaning_vi": "Chân núi. Như Thái Sơn chi lộc [泰山之麓] chân núi Thái Sơn", "meo": "", "reading": "ふもと", "vocab": [("麓", "ふもと", "chân núi")]},
    "薦": {"viet": "TIẾN", "meaning_vi": "Cỏ, rơm cho súc vật ăn gọi là tiến.", "meo": "", "reading": "すす.める", "vocab": [("薦め", "すすめ", "sự giới thiệu"), ("他薦", "たせん", "sự giới thiệu; sự tiến cử")]},
    "慶": {"viet": "KHÁNH, KHƯƠNG, KHANH", "meaning_vi": "Mừng. Như tục gọi chúc thọ là xưng khánh [稱慶].", "meo": "", "reading": "よろこ.び", "vocab": [("慶び", "よろこび", "sự sung sướng vô ngần"), ("慶事", "けいじ", "điềm lành; điềm tốt; sự kiện đáng mừng")]},
    "撤": {"viet": "TRIỆT", "meaning_vi": "Bỏ đi, trừ đi, cất đi. Như triệt hồi [撤回] rút về.", "meo": "", "reading": "テツ", "vocab": [("撤兵", "てっぺい", "sự lui binh; sự rút binh"), ("撤去", "てっきょ", "sự hủy bỏ; sự bãi bỏ")]},
    "棄": {"viet": "KHÍ", "meaning_vi": "vứt bỏ, từ bỏ", "meo": "Vứt bỏ cây (木) và dao (匕) vào cái hộp (罙)", "reading": "き", "vocab": [("放棄", "ほうき", "từ bỏ, vứt bỏ"), ("破棄", "はき", "phá hủy, hủy bỏ")]},
    "叔": {"viet": "THÚC", "meaning_vi": "Bé, anh gọi em là thúc. Như nhị thúc [二叔] chú hai.", "meo": "", "reading": "シュク", "vocab": [("叔母", "おば", "cô"), ("叔母", "しゅくぼ", "dì .")]},
    "淑": {"viet": "THỤC", "meaning_vi": "Trong trẻo, hiền lành.", "meo": "", "reading": "しと.やか", "vocab": [("淑女", "しゅくじょ", "cô; bà"), ("淑徳", "しゅくとく", "đức tính tốt của người phụ nữ; lão nhân đức độ cao; thục đức .")]},
    "寂": {"viet": "TỊCH", "meaning_vi": "Lặng yên. Như tịch mịch [寂寞]. Đỗ Phủ [杜甫] : Ngư long tịch mịch thu giang lãnh, Cố quốc bình cư hữu sở tư [魚龍寂寞秋江冷, 故國平居有所思] (Thu hứng [秋興]). Quách Tấn dịch thơ : Cá rồng vắng vẻ sông thu lạnh, Thong thả lòng thêm nhớ cố hương.", "meo": "", "reading": "さび さび.しい さび.れる さみ.しい", "vocab": [("寂", "さび", "sự tĩnh mịch; sự lẻ loi; sự buồn bã"), ("入寂", "にゅうじゃく", "Sự nhập tịch (chết) của nhà sư; nát bàn; sự tự do tinh thần .")]},
    "督": {"viet": "ĐỐC", "meaning_vi": "Đốc suất, lấy thân đốc suất kẻ dưới gọi là đốc.", "meo": "", "reading": "トク", "vocab": [("督促", "とくそく", "sự đốc thúc; sự thúc giục"), ("督励", "とくれい", "sự cổ vũ; sự khuyến khích")]},
    "戚": {"viet": "THÍCH", "meaning_vi": "Thương. Như ai thích chi dong [哀戚之容] cái dáng thương xót.", "meo": "", "reading": "いた.む うれ.える みうち", "vocab": [("休戚", "きゅうせき", "hạnh phúc; phúc lợi"), ("姻戚", "いんせき", "mối quan hệ")]},
    "朴": {"viet": "PHÁC, BỐC", "meaning_vi": "Cây phác, vỏ nó dùng làm thuốc được gọi là hậu phác [厚朴]. Thứ mọc ở tỉnh Tứ Xuyên thì tốt hơn, nên gọi là xuyên phác [川朴].", "meo": "", "reading": "ほう ほお えのき", "vocab": [("惇朴", "あつしほう", "đơn"), ("朴とつ", "ぼくとつ", "thật")]},
    "赴": {"viet": "PHÓ", "meaning_vi": "Chạy tới, tới chỗ đã định tới gọi là phó. Như bôn phó [奔赴] chạy tới.", "meo": "", "reading": "おもむ.く", "vocab": [("赴く", "おもむく", "tới; đến; đi về phía; xu hướng; phát triển theo hướng"), ("赴任", "ふにん", "việc tới nhận chức")]},
    "訃": {"viet": "PHÓ", "meaning_vi": "Báo tin có tang. Ta gọi là cáo phó [告訃].", "meo": "", "reading": "しらせ", "vocab": [("訃報", "ふほう", "báo tang")]},
    "豹": {"viet": "BÁO", "meaning_vi": "Con báo (con beo). Thứ báo có vằn như đồng tiền vàng gọi là kim tiền báo [金錢豹]. Nguyễn Du [阮攸] : Giản vụ tự sinh nghi ẩn báo [澗霧自生宜隱豹] (Đông A sơn lộ hành [東阿山路行]) Sương móc bốc lên hợp cho con báo ẩn nấp.", "meo": "", "reading": "ヒョウ ホウ", "vocab": [("豹", "ひょう", "báo"), ("全豹", "ぜんぴょう", "điềm")]},
    "貌": {"viet": "MẠO, MỘC", "meaning_vi": "Dáng mặt. Như tuyết phu hoa mạo [雪膚花貌] da như tuyết, mặt như hoa.", "meo": "", "reading": "かたち かたどる", "vocab": [("体貌", "たいぼう", "sự xuất hiện"), ("変貌", "へんぼう", "sự biến hình; sự biến dạng")]},
    "墾": {"viet": "KHẨN", "meaning_vi": "Khai khẩn, dùng sức vỡ các ruộng hoang ra mà cầy cấy gọi là khẩn.", "meo": "", "reading": "コン", "vocab": [("未墾", "みこん", "không cày cấy; bỏ hoang"), ("墾田", "こんでん", "ruộng lúa mới .")]},
    "懇": {"viet": "KHẨN", "meaning_vi": "Khẩn khoản.", "meo": "", "reading": "ねんご.ろ", "vocab": [("懇ろ", "ねんごろ", "lịch sự; nhã nhặn; hiếu khách; mến khách"), ("懇切", "こんせつ", "chi tiết; nhiệt tình; tận tâm")]},
    "龍": {"viet": "LONG, SỦNG", "meaning_vi": "Con rồng.", "meo": "", "reading": "たつ", "vocab": [("龍", "りゅう", "con rồng"), ("土龍", "どりゅう", "đê chắn sóng")]},
    "籠": {"viet": "LUNG, LỘNG", "meaning_vi": "Cái lồng đan bằng tre để đựng đồ hay đậy đồ.", "meo": "", "reading": "かご こ.める こも.る こ.む", "vocab": [("籠", "かご", "giỏ; cái giỏ; cái lồng"), ("籠城", "かごじょう", "sự bao vây")]},
    "襲": {"viet": "TẬP", "meaning_vi": "Áo mặc chồng ra ngoài. Một bộ quần áo gọi là nhất tập [一襲]. Lễ Ký [禮記] : Hàn bất cảm tập [寒不敢襲] (Nội tắc [內則]) Lạnh không dám mặc thêm áo ngoài.", "meo": "", "reading": "おそ.う かさ.ね", "vocab": [("襲う", "おそう", "công kích; tấn công ."), ("世襲", "せしゅう", "sự di truyền; tài sản kế thừa .")]},
    "寵": {"viet": "SỦNG", "meaning_vi": "Yêu, ân huệ, vẻ vang.", "meo": "", "reading": "めぐ.み めぐ.む", "vocab": [("寵児", "ちょうじ", "con yêu; đứa con được yêu chiều ."), ("寵姫", "ちょうき", "vợ yêu; thiếp yêu .")]},
    "聾": {"viet": "LUNG", "meaning_vi": "Điếc.", "meo": "", "reading": "ろう.する つんぼ みみしい", "vocab": [("聾", "つんぼ", "Sự điếc; kẻ điếc"), ("聾唖", "ろうあ", "câm điếc")]},
    "微": {"viet": "VI", "meaning_vi": "Mầu nhiệm. Như tinh vi [精微], vi diệu [微妙] nghĩa là tinh tế mầu nhiệm không thể nghĩ bàn được.", "meo": "", "reading": "かす.か", "vocab": [("微か", "かすか", "nhỏ bé"), ("微乳", "びにゅう", "bộ ngực nhỏ .")]},
    "徴": {"viet": "TRƯNG, CHỦY, TRỪNG", "meaning_vi": "Một dạng của chữ trưng [徵].", "meo": "", "reading": "しるし", "vocab": [("徴", "しるし", "đồng Mác"), ("徴候", "ちょうこう", "dấu")]},
    "懲": {"viet": "TRỪNG", "meaning_vi": "Răn bảo, trừng trị. Răn bảo cho biết sợ không dám làm bậy nữa gọi là trừng. Như bạc trừng [薄懲] trừng trị qua, nghiêm trừng [嚴懲] trừng trị nặng.", "meo": "", "reading": "こ.りる こ.らす こ.らしめる", "vocab": [("懲役", "ちょうえき", "phạt tù cải tạo"), ("懲悪", "ちょうあく", "sự trừng phạt cái ác .")]},
    "烏": {"viet": "Ô", "meaning_vi": "Con quạ, quạ con biết mớm quạ già cho nên sự hiếu dưỡng cha mẹ gọi là ô điểu chi tư [烏鳥之私].", "meo": "", "reading": "からす いずくんぞ なんぞ", "vocab": [("烏", "からす", "quạ"), ("旅烏", "たびがらす", "người đi lang thang")]},
    "鳩": {"viet": "CƯU", "meaning_vi": "Con tu hú. Tính nó vụng không biết làm tổ, nên hay dùng để nói ví những kẻ không biết kinh doanh việc nhà. Nó lại là một loài chim ăn không mắc nghẹn bao giờ, cho nên những gậy của người già chống hay khắc hình con cưu vào. Như cưu trượng [鳩杖] gậy khắc hình chim cưu.", "meo": "", "reading": "はと あつ.める", "vocab": [("鳩", "はと", "bồ câu"), ("鳩信", "きゅうしん", "việc trao đổi thông tin nhờ bồ câu đưa thư; bồ câu đưa thư .")]},
    "鴨": {"viet": "ÁP", "meaning_vi": "Con vịt.", "meo": "", "reading": "かも あひる", "vocab": [("鴨", "かも", "vịt rừng; vịt trời; kẻ ngốc nghếch dễ bị đánh lừa"), ("合鴨", "あいがも", "Sự lai giống giữa vịt trời và vịt nhà .")]},
    "鵜": {"viet": "ĐỀ", "meaning_vi": "Đề hồ [鵜鶘] một thứ chim ở nước, lông màu đỏ, đầu nhỏ, mỏ dài, dưới hàm có cái túi, bắt được cá thì đựng ở cái túi ấy. Tục gọi là đào hà [淘河] có lẽ là con bồ nông. Cũng viết là đào nga [淘鵝]. Còn có tên là già lam điểu [伽藍鳥].", "meo": "", "reading": "う", "vocab": [("鵜", "う", "chim cốc"), ("鵜飼薬", "うがいやく", "thuốc súc họng; thuốc xúc miệng")]},
    "鶴": {"viet": "HẠC", "meaning_vi": "Chim hạc, sếu. Nguyễn Trãi [阮薦] : Viên hạc tiêu điều ý phỉ câm [猿鶴蕭條意匪禁] (Khất nhân họa Côn Sơn đồ [乞人畫崑山圖]) Vượn và hạc tiêu điều, cảm xúc khó cầm.", "meo": "", "reading": "つる", "vocab": [("鶴", "つる", "con hạc; con sếu"), ("鶴嘴", "つるはし", "Cuốc chim .")]},
    "鶏": {"viet": "KÊ", "meaning_vi": "Con gà", "meo": "", "reading": "にわとり とり", "vocab": [("鶏", "にわとり", "gà ."), ("鶏冠", "けいかん", "mào gà")]},
    "鷹": {"viet": "ƯNG", "meaning_vi": "Chim ưng, con cắt, giống chim rất mạnh, chuyên bắt các chim khác ăn thịt, người đi săn thường nuôi nó để săn các chim khác. Nguyễn Du [阮攸] : Cao nguyên phong thảo hô ưng lộ [高原豐草呼鷹路] (Hàm Đan tức sự [邯鄲即事]) Bãi cỏ tươi xanh trên cao nguyên là đường gọi chim ưng (đi săn).", "meo": "", "reading": "たか", "vocab": [("鷹", "たか", "chim ưng"), ("兀鷹", "はげたか", "Chim kền kền .")]},
    "鷲": {"viet": "THỨU", "meaning_vi": "Kên kên, một giống chim hung tợn.", "meo": "", "reading": "シュウ ジュ", "vocab": [("鷲", "わし", "đại bàng ."), ("犬鷲", "いぬわし", "chim ưng vàng")]},
    "耕": {"viet": "CANH", "meaning_vi": "Cầy ruộng. Như canh tác [耕作] cầy cấy, chỉ việc làm ruộng.", "meo": "", "reading": "たがや.す", "vocab": [("耕す", "たがやす", "bưởi"), ("耕作", "こうさく", "canh tác")]},
    "耗": {"viet": "HÁO, MẠO, MAO, HAO", "meaning_vi": "Hao sút. Như háo phí ngân tiền [耗費銀錢] hao phí tiền bạc.", "meo": "", "reading": "モウ コウ カウ", "vocab": [("損耗", "そんもう", "sự mất; sự thua lỗ ."), ("摩耗", "まもう", "sự mang; sự dùng; sự mặc")]},
    "彗": {"viet": "TUỆ", "meaning_vi": "Cái chổi.", "meo": "", "reading": "ほうき", "vocab": [("彗星", "すいせい", "sao chổi")]},
    "乙": {"viet": "ẤT", "meaning_vi": "Can Ất, can thứ hai trong mười can.", "meo": "", "reading": "おと- きのと", "vocab": [("乙", "おつ", "Ất (can chi); dí dỏm; lộng lẫy; hớn hở"), ("乙", "きのと", "Ất (can); bên B (hợp đồng)")]},
    "乞": {"viet": "KHẤT, KHÍ", "meaning_vi": "Xin. Như khất thực [乞食] xin ăn. Nguyễn Du [阮攸 ] : Vân thị thành ngoại lão khất tử [云是城外老乞子] (Thái Bình mại ca giả [太平賣歌者]) Nói rằng đó là ông lão ăn mày ở ngoại thành.", "meo": "", "reading": "こ.う", "vocab": [("乞い", "こい", "lời thỉnh cầu"), ("乞う", "こう", "hỏi")]},
    "迄": {"viet": "HẤT", "meaning_vi": "Đến. Như hất kim [迄今] đến nay (kể từ trước đến nay). Nguyễn Du [阮攸] : Thử sự hất kim dĩ kinh cổ [此事迄今已經古] (Kỳ lân mộ [騏麟墓]) Việc đó đến nay đã lâu rồi.", "meo": "", "reading": "まで およ.ぶ", "vocab": [("迄", "まで", "cho đến"), ("迄に", "までに", "gần")]},
    "吃": {"viet": "CẬT", "meaning_vi": "Nói lắp.", "meo": "", "reading": "ども.る", "vocab": [("吃り", "きつり", "bệnh cà lăm"), ("吃る", "きつる", "cà lăm")]},
    "又": {"viet": "HỰU", "meaning_vi": "Lại.", "meo": "", "reading": "また また- また.の-", "vocab": [("又", "また", "lại"), ("又々", "またまた", "lại; lại một lần nữa .")]},
    "桑": {"viet": "TANG", "meaning_vi": "Cây dâu, lá dùng để chăn tằm, quả chín ăn ngon gọi là tang thẩm [桑葚].", "meo": "", "reading": "くわ", "vocab": [("桑", "くわ", "dâu tằm; dâu"), ("桑園", "そうえん", "dâu tằm .")]},
    "綴": {"viet": "CHUẾ, CHUYẾT, XUYẾT", "meaning_vi": "Nối liền, khíu liền, khâu lại.", "meo": "", "reading": "と.じる つづ.る つづり すみ.やか", "vocab": [("綴", "つづり", "sự viết vần"), ("綴り", "つづり", "sự đánh vần")]},
    "鱗": {"viet": "LÂN", "meaning_vi": "Vẩy cá. Tô Thức [蘇軾] : Cử võng đắc ngư, cự khẩu tế lân [舉網得魚, 巨口細鱗] (Hậu Xích Bích phú [後赤壁賦 ]) Cất lưới được cá, miệng to vẩy nhỏ.", "meo": "", "reading": "うろこ こけ こけら", "vocab": [("鱗", "うろこ", "vảy"), ("鱗状", "りんじょう", "có vảy; xếp như vảy cá")]},
    "傑": {"viet": "KIỆT", "meaning_vi": "Giỏi lạ. Trí khôn gấp mười người gọi là kiệt. Như hào kiệt chi sĩ [豪傑之士] kẻ sĩ hào kiệt. Nguyễn Trãi [阮廌] : Hào kiệt công danh thử địa tằng [豪傑功名此地曾] (Bạch Đằng hải khẩu [白藤海口]) Hào kiệt đã từng lập công danh ở đất này.", "meo": "", "reading": "すぐ.れる", "vocab": [("傑", "けつ", "sự ưu tú; sự xuất sắc; sự giỏi giang hơn người ."), ("人傑", "じんけつ", "người anh hùng")]},
    "瞬": {"viet": "THUẤN", "meaning_vi": "Nháy mắt.", "meo": "", "reading": "またた.く まじろ.ぐ", "vocab": [("瞬き", "まばたき", "sự nháy mắt"), ("瞬く", "またたく", "nhấp nháy")]},
    "幣": {"viet": "TỆ", "meaning_vi": "Lụa, đời xưa thường dùng làm đồ tặng nhau.", "meo": "", "reading": "ぬさ", "vocab": [("幣制", "へいせい", "chế độ tiền tệ"), ("奉幣", "ほうへい", "pháo")]},
    "弊": {"viet": "TỆ, TẾ", "meaning_vi": "Xấu, hại, rách. Như lợi tệ [利弊] lợi hại, tệ bố [弊布] giẻ rách, v.v.", "meo": "", "reading": "ヘイ", "vocab": [("余弊", "よへい", "người ở lại  sau khi hết nhiệm kỳ"), ("党弊", "とうへい", "tệ nạn trong Đảng; sự xấu xa của Đảng .")]},
    "蔽": {"viet": "TẾ", "meaning_vi": "Che đậy, che giấu", "meo": "Cỏ (艹) đội trên đầu quan (官) để che (敝) cái gì đó.", "reading": "hei", "vocab": [("遮蔽", "shahei", "Che chắn, che đậy"), ("隠蔽", "inpei", "Ẩn giấu, che giấu")]},
    "瞥": {"viet": "MIẾT", "meaning_vi": "Liếc qua.", "meo": "", "reading": "ベツ ヘツ", "vocab": [("一瞥", "いちべつ", "quặng bóng")]},
    "脛": {"viet": "HĨNH", "meaning_vi": "Cẳng chân, từ đầu gối đến chân gọi là hĩnh. Nguyễn Du [阮攸] : Tính thành hạc hĩnh hà dung đoạn [性成鶴脛何容斷] (Tự thán [自嘆]) Chân hạc tánh vốn dài, cắt ngắn làm sao được.", "meo": "", "reading": "すね はぎ", "vocab": [("脛", "すね", "cẳng chân; ống quyển (cẳng chân)"), ("脛巾", "はばき", "xà cạp")]},
    "巡": {"viet": "TUẦN", "meaning_vi": "Đi tuần, đi xem xét khu đất mình cai trị gọi là tuần.", "meo": "", "reading": "めぐ.る めぐ.り", "vocab": [("巡る", "めぐる", "đi quanh; dạo quanh"), ("一巡", "いちじゅん", "sự đập; tiếng đập")]},
    "拶": {"viet": "TẠT", "meaning_vi": "Bức bách (đè ép).", "meo": "", "reading": "せま.る", "vocab": [("挨拶", "あいさつ", "lời chào; sự chào hỏi"), ("挨拶する", "あいさつ", "chào; chào hỏi")]},
    "璃": {"viet": "LI", "meaning_vi": "Lưu ly, thủy tinh", "meo": "Vua (王) làm việc nhà (家) giúp Lý (リ)", "reading": "リ", "vocab": [("瑠璃", "るり", "Lưu ly"), ("光璃", "ひかり", "Ánh sáng lung linh (thường dùng trong tên)")]},
    "禽": {"viet": "CẦM", "meaning_vi": "Loài chim. Như gia cầm [家禽] chim gà nuôi trong nhà.", "meo": "", "reading": "とり とりこ", "vocab": [("家禽", "かきん", "Gia cầm"), ("水禽", "すいきん", "chim ở nước (mòng két")]},
    "檎": {"viet": "CẦM", "meaning_vi": "Lâm cầm [林檎] cây làm cầm, một thứ cây ăn quả, tục gọi là hoa hồng [花紅] hay sa quả [沙果].", "meo": "", "reading": "キン ゴン ゴ", "vocab": [("林檎", "りんご", "táo; quả táo .")]},
    "雛": {"viet": "SỒ", "meaning_vi": "Non. Chim non gọi là sồ, gà con cũng gọi là sồ, trẻ con cũng gọi là sồ.", "meo": "", "reading": "ひな ひよこ", "vocab": [("雛", "ひな", "gà con"), ("雛", "ひよこ", "gà con")]},
    "皺": {"viet": "TRỨU", "meaning_vi": "Mặt nhăn, nhăn nhó. Như mãn kiểm trứu văn [滿臉皺紋] mặt đầy nếp nhăn.", "meo": "", "reading": "しわ しぼ", "vocab": [("皺", "しわ", "nếp nhăn; nếp gấp ."), ("皺くちゃ", "しわくちゃ", "nhăn")]},
    "爽": {"viet": "SẢNG", "meaning_vi": "Sáng, trời sắp sáng gọi là muội sảng [昧爽].", "meo": "", "reading": "あき.らか さわ.やか たがう", "vocab": [("爽快", "そうかい", "làm cho khoẻ khoắn"), ("爽やか", "さわやか", "dễ chịu; sảng khoái")]},
    "璽": {"viet": "TỈ", "meaning_vi": "Cái ấn của thiên tử.", "meo": "", "reading": "ジ", "vocab": [("璽", "じ", "triện của vua ."), ("印璽", "いんじ", "sao lại")]},
    "鬱": {"viet": "ÚC, UẤT", "meaning_vi": "Uất kết, uất tức, khí nó tụ không tan ra, hở ra gọi là uất. Như uất kết [鬱結] uất ức, uất muộn [鬱悶] bậm bực, v.v.", "meo": "", "reading": "うっ.する ふさ.ぐ しげ.る", "vocab": [("鬱々", "うつうつ", "cảnh tối tăm"), ("鬱屈", "うっくつ", "tối tăm")]},
    "漆": {"viet": "TẤT, THẾ", "meaning_vi": "Sông Tất.", "meo": "", "reading": "うるし", "vocab": [("漆", "うるし", "cây sơn"), ("仮漆", "かりうるし", "véc ni")]},
    "膝": {"viet": "TẤT", "meaning_vi": "Đầu gối.", "meo": "", "reading": "ひざ", "vocab": [("膝", "ひざ", "đầu gối ."), ("小膝", "こひざ", "đầu gối")]},
    "黎": {"viet": "LÊ", "meaning_vi": "Đen. Bách tính, dân chúng gọi là lê dân [黎民] nghĩa là kể số người tóc đen vậy. Cũng gọi là lê nguyên [黎元].", "meo": "", "reading": "くろ.い", "vocab": [("黎明", "れいめい", "lúc tảng sáng"), ("黎明期", "れいめいき", "bình minh")]},
    "姦": {"viet": "GIAN", "meaning_vi": "Gian giảo. Như chữ gian [奸].", "meo": "", "reading": "かん.する かしま.しい みだら", "vocab": [("姦人", "かんじん", "côn đồ; kẻ hung ác"), ("佞姦", "ねいかん", "Bội tín; đồi bại; hư thân mất nết .")]},
    "晶": {"viet": "TINH", "meaning_vi": "Trong suốt. Vật gì có chất sáng suốt bên nọ sang bên kia gọi là tinh oánh [晶瑩].", "meo": "", "reading": "ショウ", "vocab": [("晶化", "あきらか", "sự kết tinh"), ("水晶", "すいしょう", "pha lê")]},
    "囁": {"viet": "CHIẾP", "meaning_vi": "Chiếp nhu [囁嚅] nhập nhù. Tô Mạn Thù [蘇曼殊] : Cửu nãi chiếp nhu ngôn viết [久乃囁嚅言曰] (Đoạn hồng linh nhạn kí [斷鴻零雁記]) Một chặp lâu sau mới ấp úng mà bảo rằng.", "meo": "", "reading": "ささや.く", "vocab": [("囁き", "ささやき", "tiếng nói thầm"), ("囁く", "ささやく", "xào xạc; róc rách; thì thầm; xì xào .")]},
    "琴": {"viet": "CẦM", "meaning_vi": "Cái đàn cầm. Đàn dài ba thước sáu tấc, căng bảy dây gọi là đàn cầm. Nguyễn Trãi [阮薦] : Giai khách tương phùng nhật bão cầm [佳客相逢日抱琴] (Đề Trình xử sĩ Vân oa đồ [題程處士雲窩圖]) Khách quý gặp nhau, ngày ngày ôm đàn gảy.", "meo": "", "reading": "こと", "vocab": [("琴", "こと", "đàn Koto"), ("提琴", "ていきん", "đàn viôlông")]},
    "琵": {"viet": "TÌ", "meaning_vi": "Tì bà [琵琶] cái đàn tì bà có bốn dây.", "meo": "", "reading": "ビ ヒ", "vocab": [("琵琶", "びわ", "Biwa; cái đàn luýt nhật .")]},
    "琶": {"viet": "BÀ", "meaning_vi": "Xem tì bà [琵琶].", "meo": "", "reading": "ハ ベ ワ", "vocab": [("琵琶", "びわ", "Biwa; cái đàn luýt nhật .")]},
    "馳": {"viet": "TRÌ", "meaning_vi": "Rong ruổi, tả cái dáng ngựa chạy nhanh. Tô Thức [蘇軾] : Trì sính đương thế [馳騁當世] (Phương Sơn Tử truyện [方山子傳]) Rong ruổi ở đời.", "meo": "", "reading": "は.せる", "vocab": [("背馳", "はいち", "sự mâu thuẫn"), ("馳走", "ちそう", "sự đãi")]},
    "唸": {"viet": "NIỆM", "meaning_vi": "groan, roar", "meo": "", "reading": "うな.る うなり", "vocab": [("唸り", "うなり", "tiếng rền rĩ; tiếng hú; tiếng gầm rú; sự rền rĩ; sự gầm rú; tiếng kêu ."), ("唸る", "うなる", "kêu rú; hú; kêu rống; rền rĩ; gầm; sủa; kêu; gầm gừ; cằn nhằn; rên rỉ .")]},
    "吠": {"viet": "PHỆ", "meaning_vi": "sủa, hú", "meo": "Chữ KHUYỂN đứng cạnh bộ KHẨU, con chó há mồm ra sủa", "reading": "ほ・える", "vocab": [("吠える", "ほえる", "sủa, hú"), ("遠吠え", "とおぼえ", "tiếng hú xa xăm")]},
    "贅": {"viet": "CHUẾ", "meaning_vi": "Khíu lại, bám vào. Như chuế vưu [贅肬] cái bướu mọc ở ngoài da, vì thế nên vật gì thừa, vô dụng cũng gọi là chuế.", "meo": "", "reading": "いぼ", "vocab": [("贅沢", "ぜいたく", "sự xa xỉ"), ("贅肉", "ぜいにく", "tình trạng mềm nhão cơ bắp ở người")]},
    "痒": {"viet": "DƯƠNG, DƯỠNG, DẠNG", "meaning_vi": "Ốm.", "meo": "", "reading": "かゆ.がる かさ かゆ.い", "vocab": [("痒い", "かゆい", "ngứa; ngứa rát"), ("掻痒", "そうよう", "sự ngứa; bệnh ngứa; bệnh ghẻ")]},
    "掻": {"viet": "TAO", "meaning_vi": "cào, gãi, khuấy", "meo": "Tay (扌) nắm cái cào (蚤) để cào.", "reading": "か", "vocab": [("掻く", "かく", "cào, gãi"), ("掻き回す", "かきまわす", "khuấy động, đảo lộn")]},
}

def get_kanji_info(kanji: str, meo: str = "") -> dict:

    """

    Lấy thông tin đầy đủ cho 1 kanji.

    Ưu tiên: MANUAL_VI → N4_VI → MNN_N4_EXTRA → MNN_N5 → Cache → Gemini/Jisho API.

    """

    info = {

        "kanji": kanji,

        "viet": "",

        "meaning_vi": "",

        "meo": meo,

        "reading": "",

        "meanings_en": [],

        "vocab": [],

    }



    # ── DB nội bộ (tức thì, không cần cache) ──

    for db in [MANUAL_VI, N4_VI, MNN_N4_EXTRA, MNN_N5, N3_VI, N2_VI, N1_VI]:

        if kanji in db:

            info.update(db[kanji])

            if meo:

                info["meo"] = meo

            return info



    # ── Cache đĩa ──

    if kanji in _mem_cache:

        cached = _mem_cache[kanji].copy()

        cached["kanji"] = kanji

        if meo:

            cached["meo"] = meo

        return cached



    # Fallback 1: Mazii Dictionary (Ưu tiên hàng đầu cho tiếng Việt)

    mazii = lookup_kanji_mazii(kanji)

    if mazii:

        info.update(mazii)

        if meo: info["meo"] = meo

        with _cache_lock:

            _mem_cache[kanji] = {k: v for k, v in info.items() if k != "kanji"}

        threading.Thread(target=_save_cache, daemon=True).start()

        return info



    # Fallback 2: AI (Gemini hoặc OpenRouter tùy cấu hình)

    provider = get_ai_provider()

    ai_info = None

    if provider == "gemini":

        ai_info = lookup_kanji_gemini(kanji)

        # Tự động failover sang OpenRouter nếu Gemini lỗi (nếu có key)

        if not ai_info and get_openrouter_key():

            ai_info = lookup_kanji_openrouter(kanji)

            if ai_info: info["source"] = "openrouter"

    else:

        ai_info = lookup_kanji_openrouter(kanji)

        if ai_info: info["source"] = "openrouter"



    if ai_info:

        info.update(ai_info)

        if meo:

            info["meo"] = meo

        if "source" not in info:

            info["source"] = "gemini"

        with _cache_lock:

            _mem_cache[kanji] = {k: v for k, v in info.items() if k != "kanji"}

        threading.Thread(target=_save_cache, daemon=True).start()

        return info



    # Fallback 2: Jisho API

    jisho = lookup_kanji_jisho(kanji)

    info["reading"] = jisho.get("reading", "")

    info["meanings_en"] = jisho.get("meanings_en", [])

    info["meaning_vi"] = jisho.get("meaning_vi") or (", ".join(info["meanings_en"]) if info["meanings_en"] else kanji)

    info["source"] = "jisho"

    with _cache_lock:

        _mem_cache[kanji] = {k: v for k, v in info.items() if k != "kanji"}

    threading.Thread(target=_save_cache, daemon=True).start()

    return info





def search_by_viet(query: str) -> list[dict]:

    """

    Tìm kiếm kanji theo âm Hán Việt hoặc nghĩa tiếng Việt.

    Khớp chính xác case-insensitive, giữ nguyên dấu thanh:

      "Tinh" ≠ "Tĩnh", "TĨNH" == "tĩnh"

    Trả về list các info dict khớp (tối đa 20 kết quả).

    """

    query = query.strip()

    if not query:

        return []



    q_lower = query.lower()

    results: list[dict] = []

    seen: set = set()



    for db in [MANUAL_VI, N4_VI, MNN_N4_EXTRA, MNN_N5, N3_VI, N2_VI, N1_VI]:

        for kanji, data in db.items():

            if kanji in seen:

                continue

            if (q_lower in data.get("viet", "").lower()

                    or q_lower in data.get("meaning_vi", "").lower()

                    or q_lower in data.get("meo", "").lower()):

                results.append({"kanji": kanji, "vocab": [], **data})

                seen.add(kanji)

        if len(results) >= 20:

            break



    return results[:20]





_GEMINI_REVERSE_PROMPT = """

Bạn là chuyên gia dạy tiếng Nhật cho người Việt Nam.

Người dùng đang tìm kanji với từ khóa: "{query}"

Từ khóa này có thể là âm Hán Việt (VD: "SƠN", "NHẬT"), nghĩa tiếng Việt (VD: "núi", "học"), hoặc kết hợp.



Hãy liệt kê TỐI ĐA 5 kanji phù hợp nhất và trả về JSON sau (KHÔNG giải thích thêm):

[

  {{

    "kanji": "字",

    "viet": "ÂM HÁN VIỆT",

    "reading": "hiragana phổ biến nhất",

    "meaning_vi": "nghĩa tiếng Việt ngắn gọn",

    "meo": "mẹo nhớ vui bằng tiếng Việt (để trống nếu không có)",

    "vocab": [

      ["từ_kanji_1", "hiragana_1", "nghĩa_việt_1"],

      ["từ_kanji_2", "hiragana_2", "nghĩa_việt_2"]

    ]

  }}

]

"""





def search_by_viet_gemini(query: str) -> list[dict]:

    """Dùng Gemini để tìm kanji theo âm HV / nghĩa tiếng Việt khi DB không có kết quả."""

    api_key = get_gemini_key()

    if not api_key:

        return []

    try:

        from google import genai

        client = genai.Client(api_key=api_key)

        prompt = _GEMINI_REVERSE_PROMPT.format(query=query)

        response = client.models.generate_content(

            model=GEMINI_MODEL,

            contents=prompt,

        )

        raw = response.text.strip()

        if raw.startswith("```"):

            raw = raw.split("\n", 1)[-1]

            raw = raw.rsplit("```", 1)[0]

        items = json.loads(raw)

        results = []

        for item in items[:5]:

            vocab_raw = item.get("vocab", [])

            vocab = [tuple(v) for v in vocab_raw if len(v) >= 3]

            info = {

                "kanji":      item.get("kanji", ""),

                "viet":       item.get("viet", ""),

                "reading":    item.get("reading", ""),

                "meaning_vi": item.get("meaning_vi", ""),

                "meo":        item.get("meo", ""),

                "vocab":      vocab[:2],

                "meanings_en": [],

                "source":     "gemini",

            }

            if info["kanji"]:

                results.append(info)

        return results

    except Exception as e:

        msg = str(e)

        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:

            raise GeminiQuotaError("Hết quota Gemini hôm nay, vui lòng thử lại vào ngày mai.")

        print(f"[gemini-reverse] Lỗi: {e}")

        return []





def search_by_viet_openrouter(query: str) -> list[dict]:

    """Dùng OpenRouter để tìm kanji theo âm HV / nghĩa tiếng Việt."""

    api_key = get_openrouter_key()

    if not api_key:

        return []

        

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {

        "Authorization": f"Bearer {api_key}",

        "Content-Type": "application/json",

    }

    

    prompt = _GEMINI_REVERSE_PROMPT.format(query=query)

    payload = {

        "model": OPENROUTER_MODEL,

        "messages": [{"role": "user", "content": prompt}],

        "response_format": {"type": "json_object"}

    }

    

    try:

        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.encoding = 'utf-8'
        resp_data = resp.json()
        raw = resp_data["choices"][0]["message"]["content"].strip()

        

        if raw.startswith("```"):

            raw = raw.split("\n", 1)[-1]

            raw = raw.rsplit("```", 1)[0]

            

        # Một số model trả về JSON trực tiếp, một số trả về bọc trong key

        items_data = json.loads(raw)

        if isinstance(items_data, dict):

            # Tìm key chứa list (thường là model trả về dict { "results": [...] })

            for v in items_data.values():

                if isinstance(v, list):

                    items = v

                    break

            else:

                items = []

        else:

            items = items_data



        results = []

        for item in items[:5]:

            vocab_raw = item.get("vocab", [])

            vocab = [tuple(v) for v in vocab_raw if len(v) >= 3]

            info = {

                "kanji":      item.get("kanji", ""),

                "viet":       item.get("viet", ""),

                "reading":    item.get("reading", ""),

                "meaning_vi": item.get("meaning_vi", ""),

                "meo":        item.get("meo", ""),

                "vocab":      vocab[:2],

                "meanings_en": [],

                "source":     "gemini", # Vẫn để gemini để UI hiện màu tím cho đẹp

            }

            if info["kanji"]:

                results.append(info)

        return results

    except Exception as e:

        print(f"[openrouter-reverse] Lỗi: {e}")

        return []





# ─── Vocab Lookup (từ vựng phức hợp) ────────────────────────────────────────



_VOCAB_PROMPT = """

Bạn là từ điển Nhật-Việt chuyên nghiệp.

Tra cứu từ/cụm từ tiếng Nhật "{word}" và trả về JSON CHÍNH XÁC sau (không giải thích thêm):

{{

  "word": "từ tiếng Nhật chính xác",

  "reading": "cách đọc hiragana/katakana đầy đủ",

  "han_viet": "ÂM HÁN VIỆT nếu có (để trống chuỗi rỗng nếu không có)",

  "meanings_vi": ["nghĩa tiếng Việt 1", "nghĩa tiếng Việt 2"],

  "examples": [

    {{"sentence": "Câu ví dụ tiếng Nhật 1", "reading": "phiên âm hiragana", "meaning": "dịch nghĩa tiếng Việt"}},

    {{"sentence": "Câu ví dụ tiếng Nhật 2", "reading": "phiên âm hiragana", "meaning": "dịch nghĩa tiếng Việt"}},

    {{"sentence": "Câu ví dụ tiếng Nhật 3", "reading": "phiên âm hiragana", "meaning": "dịch nghĩa tiếng Việt"}}

  ],

  "related": [

    ["từ liên quan 1", "cách đọc 1", "nghĩa việt 1"],

    ["từ liên quan 2", "cách đọc 2", "nghĩa việt 2"],

    ["từ liên quan 3", "cách đọc 3", "nghĩa việt 3"]

  ]

}}

"""






# ─── Sentence Vocab Extraction ───────────────────────────────────────────────

_SENTENCE_PROMPT = """
Ban la tu dien Nhat-Viet chuyen nghiep.
Phan tich cau tieng Nhat sau va trich xuat TOI DA 6 tu vung chinh quan trong:

Cau: "{sentence}"

Tra ve JSON array CHINH XAC (khong giai thich, khong markdown):
[
  {{"word": "tu_tieng_nhat", "reading": "hiragana", "han_viet": "AM HAN VIET hoac rong", "meanings_vi": ["nghia 1", "nghia 2"]}},
  ...
]
Chu y:
- Chi lay tu vung co nghia (danh tu, dong tu, tinh tu, trang tu quan trong)
- Bo qua tro tu, ket thuc cau (ne, yo, wa, ga, o, ni...)
- Giai thich nghia bang tieng Viet ro rang
"""


def lookup_sentence_vocab(sentence: str) -> dict:
    """Dung AI phan tich cau tieng Nhat, trich xuat tu vung quan trong."""
    import json as _json
    provider = get_ai_provider()
    api_key = get_gemini_key() if provider == "gemini" else get_openrouter_key()
    if not api_key:
        return {}

    prompt = _SENTENCE_PROMPT.format(sentence=sentence)
    raw = ""
    try:
        if provider == "gemini":
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            raw = response.text.strip()
        else:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"model": OPENROUTER_MODEL, "messages": [{"role": "user", "content": prompt}]}
            resp = requests.post(url, headers=headers, json=payload, timeout=25)
            resp.encoding = 'utf-8'
            raw = resp.json()["choices"][0]["message"]["content"].strip()

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        if "[" in raw:
            raw = raw[raw.find("["):raw.rfind("]")+1]

        items = _json.loads(raw)
        words = []
        for item in items[:6]:
            w = item.get("word", "").strip()
            if not w:
                continue
            meanings = item.get("meanings_vi", [])
            if isinstance(meanings, str):
                meanings = [meanings]
            words.append({
                "word":        w,
                "reading":     item.get("reading", ""),
                "han_viet":    item.get("han_viet", ""),
                "meanings_vi": meanings[:3],
                "examples":    [],
                "related":     [],
                "source":      provider,
            })
        if words:
            return {
                "type":     "sentence",
                "sentence": sentence,
                "words":    words,
                "source":   provider,
            }
    except Exception as e:
        print(f"[sentence-vocab] Loi: {e}")
    return {}


def _is_sentence(text: str) -> bool:
    """Kiem tra xem text co phai la cau (nhieu tu) khong."""
    if len(text) < 7:
        return False
    # Dem ky tu hiragana/katakana
    hira_kata = sum(1 for c in text if "\u3040" <= c <= "\u30ff")
    return hira_kata >= 3


def lookup_vocab_mazii(word: str) -> dict:

    """Dùng Mazii API (type=word) để tra từ vựng phức hợp."""

    try:

        url = "https://mazii.net/api/search"

        payload = {"dict": "javi", "type": "word", "query": word, "page": 1}

        resp = requests.post(url, json=payload, timeout=10)

        if resp.status_code != 200:

            return {}

        data = resp.json()

        results = data.get("results", [])

        if not results:

            return {}



        item = results[0]



        # Lấy nghĩa — "detail" chứa nghĩa đầy đủ, "mean" chứa nghĩa ngắn

        meanings = []

        detail = item.get("detail", "") or ""

        if detail:

            for m in detail.replace("##", "\n").split("\n"):

                m = m.strip()

                if m and len(m) > 1:

                    meanings.append(m)

        if not meanings:

            raw_mean = item.get("mean", "")

            if raw_mean:

                meanings = [raw_mean]



        # Lấy ví dụ từ trường examples (type=word trả về list câu hoàn chỉnh)

        examples = []

        for ex in item.get("examples", [])[:3]:

            sent = ex.get("w", "") or ex.get("e", "")

            reading_ex = ex.get("p", "").strip()

            meaning_ex = ex.get("m", "")

            if sent:

                examples.append({

                    "sentence": sent,

                    "reading":  reading_ex,

                    "meaning":  meaning_ex,

                })



        # Nếu không có examples, thử lấy từ example_on / example_kun (type=kanji)

        if not examples:

            for key in ("example_on", "example_kun"):

                ex_dict = item.get(key, {})

                if isinstance(ex_dict, dict):

                    for reading_key, ex_list in ex_dict.items():

                        for ex in ex_list[:3]:

                            sent = ex.get("w", "")

                            meaning_ex = ex.get("m", "")

                            if sent:

                                examples.append({

                                    "sentence": sent,

                                    "reading":  ex.get("p", reading_key),

                                    "meaning":  meaning_ex,

                                })

                        if len(examples) >= 3:

                            break

                if len(examples) >= 3:

                    break



        # Từ liên quan — lấy từ examples còn lại

        related = []

        for ex in item.get("examples", [])[3:7]:

            w = ex.get("w", "")

            p = ex.get("p", "").strip()

            m = ex.get("m", "")

            h = ex.get("h", "")

            if w and m:

                related.append((w, p, f"{m} ({h})" if h else m))



        # Reading: ưu tiên phonetic/kana → on → kun

        reading = (

            item.get("phonetic", "")

            or item.get("kana", "")

            or item.get("on", "")

            or item.get("kun", "")

        )



        # Âm Hán Việt — thường trong "mean" của kanji đơn, hoặc trường "h" ở examples

        han_viet = ""

        if item.get("examples"):

            han_viet = item["examples"][0].get("h", "")



        return {

            "word":        item.get("word", word) or word,

            "reading":     reading,

            "han_viet":    han_viet,

            "meanings_vi": meanings[:4],

            "examples":    examples,

            "related":     related[:4],

            "source":      "mazii",

        }

    except Exception as e:

        print(f"[mazii-vocab] Lỗi: {e}")

        return {}





def lookup_vocab_jisho(word: str) -> dict:

    """Dùng Jisho API để tra từ vựng (fallback khi Mazii không có kết quả)."""

    try:

        url = JISHO_API.format(urllib.parse.quote(word))

        resp = requests.get(url, timeout=6)

        data = resp.json()

        results = data.get("data", [])

        if not results:

            return {}



        item = results[0]

        japanese = item.get("japanese", [{}])

        word_form = japanese[0].get("word", word) if japanese else word

        reading = japanese[0].get("reading", "") if japanese else ""



        senses = item.get("senses", [])

        all_meanings_en = []

        for s in senses[:4]:

            all_meanings_en.extend(s.get("english_definitions", [])[:3])



        meanings_vi_text = translate_en_to_vi("; ".join(all_meanings_en[:5])) if all_meanings_en else ""

        meanings_vi = [meanings_vi_text] if meanings_vi_text else all_meanings_en[:3]



        # Lấy từ kết quả phụ làm related

        related = []

        for alt_item in results[1:5]:

            alt_jp = alt_item.get("japanese", [{}])

            alt_w = alt_jp[0].get("word", "") if alt_jp else ""

            alt_r = alt_jp[0].get("reading", "") if alt_jp else ""

            alt_senses = alt_item.get("senses", [{}])

            alt_m_en = alt_senses[0].get("english_definitions", []) if alt_senses else []

            alt_m = translate_en_to_vi(alt_m_en[0]) if alt_m_en else ""

            if alt_w:

                related.append((alt_w, alt_r, alt_m))



        return {

            "word":        word_form,

            "reading":     reading,

            "han_viet":    "",

            "meanings_vi": meanings_vi,

            "examples":    [],

            "related":     related,

            "source":      "jisho",

        }

    except Exception as e:

        print(f"[jisho-vocab] Lỗi: {e}")

        return {}





def lookup_vocab_ai(word: str) -> dict:

    """Dùng AI (Gemini hoặc OpenRouter) để tra từ vựng khi không tìm thấy qua API công cộng."""

    provider = get_ai_provider()

    api_key = get_gemini_key() if provider == "gemini" else get_openrouter_key()

    if not api_key:

        return {}



    prompt = _VOCAB_PROMPT.format(word=word)

    raw = ""

    try:

        if provider == "gemini":

            from google import genai

            client = genai.Client(api_key=api_key)

            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)

            raw = response.text.strip()

        else:

            url = "https://openrouter.ai/api/v1/chat/completions"

            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

            payload = {"model": OPENROUTER_MODEL, "messages": [{"role": "user", "content": prompt}]}

            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            resp.encoding = 'utf-8'
            raw = resp.json()["choices"][0]["message"]["content"].strip()



        if raw.startswith("```"):

            raw = raw.split("\n", 1)[-1]

            raw = raw.rsplit("```", 1)[0]

        if "{" in raw:

            raw = raw[raw.find("{"):raw.rfind("}")+1]



        data = json.loads(raw)



        examples = []

        for ex in data.get("examples", [])[:3]:

            if isinstance(ex, dict):

                examples.append({

                    "sentence": ex.get("sentence", ""),

                    "reading":  ex.get("reading", ""),

                    "meaning":  ex.get("meaning", ""),

                })



        related = []

        for r in data.get("related", [])[:4]:

            if isinstance(r, (list, tuple)) and len(r) >= 2:

                related.append(tuple(r[:3]))



        return {

            "word":        data.get("word", word),

            "reading":     data.get("reading", ""),

            "han_viet":    data.get("han_viet", ""),

            "meanings_vi": data.get("meanings_vi", [])[:4],

            "examples":    examples,

            "related":     related,

            "source":      provider,

        }

    except Exception as e:

        print(f"[ai-vocab] Lỗi ({provider}): {e}")

        return {}





def lookup_vocab(word: str) -> dict:
    """
    Tra cứu từ vựng đầy đủ cho một từ/cụm từ tiếng Nhật.
    Ưu ti�n: Mazii API (type=word) → Mazii API (type=kanji) → Jisho → AI
    Trả về dict với c�c key:
      word, reading, han_viet, meanings_vi, examples, related, source
    """
    if not word.strip():
        return {}

    # 1. Mazii type=word � Ưu ti�n tuyệt đối, chấp nhận kể cả khi meanings_vi rỗng
    result = lookup_vocab_mazii(word)
    if result:
        return result

    # 1b. Mazii type=kanji � thử th�m nếu l� từ ngắn
    if len(word) <= 2:
        try:
            _resp = requests.post(
                "https://mazii.net/api/search",
                json={"dict": "javi", "type": "kanji", "query": word, "page": 1},
                timeout=10,
            )
            if _resp.status_code == 200:
                _items = _resp.json().get("results", [])
                if _items:
                    _item = _items[0]
                    _detail = (_item.get("detail", "") or "")
                    _meanings = []
                    for _m in _detail.replace("##", "\n").split("\n"):
                        _m = _m.strip()
                        if _m and len(_m) > 1:
                            _meanings.append(_m)
                    if not _meanings and _item.get("mean"):
                        _meanings = [_item["mean"]]
                    _examples = []
                    for _ex in _item.get("examples", [])[:3]:
                        _w = _ex.get("w", "")
                        if _w:
                            _examples.append({
                                "sentence": _w,
                                "reading":  _ex.get("p", "").strip(),
                                "meaning":  _ex.get("m", ""),
                            })
                    if _meanings or _examples:
                        return {
                            "word":        _item.get("kanji", word),
                            "reading":     _item.get("on", "") or _item.get("kun", ""),
                            "han_viet":    _item.get("mean", ""),
                            "meanings_vi": _meanings[:4],
                            "examples":    _examples,
                            "related":     [],
                            "source":      "mazii",
                        }
        except Exception:
            pass

    # 2. Jisho � fallback khi Mazii kh�ng t�m thấy từ
    result = lookup_vocab_jisho(word)
    if result and result.get("meanings_vi"):
        # Nếu Jisho trả về từ qu� ngắn so với input (chắc l� bị cắt x�n do l� c�u d�i), bỏ qua để AI dịch cả c�u
        if len(word) >= 5 and len(result["word"]) < len(word) * 0.5:
            pass
        else:
            return result

    # 3. AI � fallback cuối c�ng, cần API key
    return lookup_vocab_ai(word)
