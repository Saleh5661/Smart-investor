# -*- coding: utf-8 -*-
"""
=================================================================
 المستثمر الذكي — واجهة برمجية (API) لتوليد استراتيجية استثمار
=================================================================

هذا الخادم يستقبل طلب POST على المسار /predict ويرجّع:
{
  "strategy": "نص الاستراتيجية المقترحة بالعربية",
  "portfolio": [ {"name": "اسم السهم", "percentage": 25}, ... ]
}

المنطق:
- يعتمد على مجموعات أسهم مصنّفة حسب القطاع (تقنية / بنوك / طاقة / الكل)،
  مقتبسة من نفس قائمة الشركات المستخدمة في مشروع نمذجة XGBoost
  (US_Engineered_Features_No_Lag_Stock.csv).
- يحاول جلب بيانات آنية عبر yfinance لترتيب الأسهم حسب الزخم
  (نسبة التغيّر خلال آخر شهر)، وإن تعذّر الاتصال بالإنترنت يستخدم
  ترتيباً افتراضياً ثابتاً بدلاً من فشل الطلب.
- يوزّع الأوزان حسب قالب جاهز لكل مستوى مخاطرة (منخفضة/متوسطة/عالية).

كيفية التشغيل محلياً:
    pip install -r requirements.txt
    python app.py
    # الخادم يعمل على http://localhost:5000

كيفية التشغيل داخل Google Colab مع ngrok:
    !pip install flask flask-cors yfinance pyngrok -q
    # ثم في خلية جديدة:
    from pyngrok import ngrok
    public_url = ngrok.connect(5000)
    print("رابط الخادم العام:", public_url)
    # ثم شغّل هذا الملف (أو الصق محتواه) وسيعمل الخادم على المنفذ 5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import datetime

# محاولة استيراد yfinance بشكل اختياري - النظام يعمل حتى بدونها
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


app = Flask(__name__)
CORS(app)  # يسمح بالطلبات من الواجهة الأمامية (متصفح المستخدم عبر ngrok)


# =================================================================
# 1) مجموعات الأسهم حسب القطاع (مقتبسة من قائمة الـ 250 شركة الأصلية)
# =================================================================
SECTOR_POOLS = {
    "tech": [
        ("AAPL", "أبل — Apple Inc."),
        ("MSFT", "مايكروسوفت — Microsoft"),
        ("NVDA", "إنفيديا — NVIDIA"),
        ("GOOGL", "ألفابت — Alphabet"),
        ("META", "ميتا — Meta Platforms"),
        ("AMD", "أدفانسد مايكرو — AMD"),
        ("CRM", "سيلزفورس — Salesforce"),
        ("ORCL", "أوراكل — Oracle"),
    ],
    "banks": [
        ("JPM", "جي بي مورغان — JPMorgan Chase"),
        ("BAC", "بنك أوف أمريكا — Bank of America"),
        ("GS", "غولدمان ساكس — Goldman Sachs"),
        ("MS", "مورغان ستانلي — Morgan Stanley"),
        ("C", "سيتي غروب — Citigroup"),
        ("SCHW", "تشارلز شواب — Charles Schwab"),
        ("V", "فيزا — Visa"),
        ("MA", "ماستركارد — Mastercard"),
    ],
    "energy": [
        ("XOM", "إكسون موبيل — ExxonMobil"),
        ("CVX", "شيفرون — Chevron"),
        ("SLB", "شلمبرجير — SLB"),
        ("COP", "كونوكو فيليبس — ConocoPhillips"),
        ("OXY", "أوكسيدنتال — Occidental"),
        ("EOG", "إي أو جي — EOG Resources"),
        ("WMB", "ويليامز — Williams Companies"),
        ("PSX", "فيليبس 66 — Phillips 66"),
    ],
    "all": [
        ("AAPL", "أبل — Apple Inc."),
        ("MSFT", "مايكروسوفت — Microsoft"),
        ("NVDA", "إنفيديا — NVIDIA"),
        ("JPM", "جي بي مورغان — JPMorgan Chase"),
        ("XOM", "إكسون موبيل — ExxonMobil"),
        ("V", "فيزا — Visa"),
        ("PG", "بروكتر آند غامبل — P&G"),
        ("KO", "كوكاكولا — Coca-Cola"),
        ("JNJ", "جونسون آند جونسون — J&J"),
        ("COP", "كونوكو فيليبس — ConocoPhillips"),
    ],
}

# نسبة نقد/صكوك افتراضية تُستخدم فقط في المخاطرة المنخفضة
CASH_ASSET_NAME = "نقد وصكوك قصيرة الأجل"

# قوالب توزيع الأوزان حسب مستوى المخاطرة (تجمع دائماً إلى 100)
RISK_WEIGHT_TEMPLATES = {
    "low":    {"include_cash": True,  "weights": [30, 25, 20, 15, 10]},  # 30% نقد + 4 أسهم
    "medium": {"include_cash": False, "weights": [30, 25, 20, 15, 10]},  # 5 أسهم متوازنة
    "high":   {"include_cash": False, "weights": [40, 30, 20, 10]},      # 4 أسهم مركّزة
}

RISK_LABELS_AR = {"low": "منخفضة", "medium": "متوسطة", "high": "عالية"}
SECTOR_LABELS_AR = {"all": "كل القطاعات", "energy": "الطاقة", "banks": "البنوك", "tech": "التقنية"}


# =================================================================
# 2) ترتيب الأسهم حسب الزخم (Momentum) باستخدام yfinance
# =================================================================
def rank_tickers_by_momentum(tickers):
    """
    يحاول جلب نسبة التغيّر خلال آخر شهر لكل سهم لترتيبها تنازلياً.
    إن تعذّر الاتصال بالإنترنت أو فشل yfinance، يُعاد الترتيب الأصلي
    دون توقف الخادم عن العمل (منطق دفاعي - Graceful Degradation).
    """
    if not YFINANCE_AVAILABLE:
        return tickers

    scored = []
    try:
        symbols = [t[0] for t in tickers]
        data = yf.download(
            symbols, period="1mo", interval="1d",
            progress=False, group_by="ticker", threads=True
        )
        for ticker, name_ar in tickers:
            try:
                closes = data[ticker]["Close"].dropna()
                if len(closes) >= 2:
                    momentum = (closes.iloc[-1] / closes.iloc[0]) - 1
                else:
                    momentum = 0.0
            except Exception:
                momentum = 0.0
            scored.append((ticker, name_ar, momentum))

        scored.sort(key=lambda x: x[2], reverse=True)
        return [(t, n) for t, n, _ in scored]

    except Exception as e:
        print(f"تنبيه: تعذّر جلب بيانات yfinance ({e}) — سيتم استخدام الترتيب الافتراضي.")
        return tickers


# =================================================================
# 3) بناء نص الاستراتيجية بالعربية
# =================================================================
def build_strategy_text(amount, risk, sector, chosen_names):
    risk_ar = RISK_LABELS_AR.get(risk, risk)
    sector_ar = SECTOR_LABELS_AR.get(sector, sector)
    assets_preview = "، ".join(chosen_names[:3])

    templates = {
        "low": (
            f"بناءً على رأس مال قدره {amount:,.0f} ريال ومستوى مخاطرة منخفض، "
            f"يقترح النموذج محفظة دفاعية تُخصّص نسبة كبيرة منها للسيولة والصكوك "
            f"لتقليل التذبذب، مع توزيع الباقي على أسهم مستقرة من قطاع {sector_ar} "
            f"مثل {assets_preview}. الهدف هنا هو الحفاظ على رأس المال أولاً، ثم تحقيق نمو تدريجي."
        ),
        "medium": (
            f"بناءً على رأس مال قدره {amount:,.0f} ريال ومستوى مخاطرة متوسط، "
            f"صمّم النموذج محفظة متوازنة موزّعة على خمسة أسهم من قطاع {sector_ar} "
            f"أبرزها {assets_preview}، بالاعتماد على مؤشرات الزخم الفني (RSI وMFI) "
            f"وتوقعات العائد لخمسة أيام تداول قادمة. هذا التوزيع يوازن بين النمو والاستقرار."
        ),
        "high": (
            f"بناءً على رأس مال قدره {amount:,.0f} ريال ومستوى مخاطرة عالٍ، "
            f"يركّز النموذج المحفظة على أربعة أسهم ذات أعلى زخم سعري في قطاع {sector_ar}، "
            f"وعلى رأسها {assets_preview}، سعياً وراء أعلى عائد ممكن خلال المدى القصير. "
            f"يُرجى ملاحظة أن هذا التوزيع المركّز يحمل تقلبات أعلى بشكل ملحوظ."
        ),
    }
    return templates.get(risk, templates["medium"])


# =================================================================
# 4) نقطة النهاية الرئيسية /predict
# =================================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        amount = data.get("amount")
        risk = data.get("risk")
        sector = data.get("sector")

        # ----- التحقق من صحة المدخلات -----
        if amount is None or not isinstance(amount, (int, float)) or amount <= 0:
            return jsonify({"error": "قيمة رأس المال يجب أن تكون رقماً أكبر من صفر"}), 400

        if risk not in RISK_WEIGHT_TEMPLATES:
            return jsonify({"error": "مستوى المخاطرة غير صالح (low / medium / high)"}), 400

        if sector not in SECTOR_POOLS:
            return jsonify({"error": "القطاع غير صالح (all / energy / banks / tech)"}), 400

        # ----- اختيار مجموعة الأسهم وترتيبها حسب الزخم -----
        pool = SECTOR_POOLS[sector]
        ranked = rank_tickers_by_momentum(pool)

        template = RISK_WEIGHT_TEMPLATES[risk]
        weights = template["weights"]
        n_stocks_needed = len(weights) - (1 if template["include_cash"] else 0)

        chosen = ranked[:n_stocks_needed]
        # في حال كانت المجموعة أصغر من المطلوب (احتياط دفاعي)
        while len(chosen) < n_stocks_needed:
            chosen.append(random.choice(pool))

        portfolio = []
        weight_idx = 0

        if template["include_cash"]:
            portfolio.append({"name": CASH_ASSET_NAME, "percentage": weights[0]})
            weight_idx = 1

        for (ticker, name_ar) in chosen:
            portfolio.append({
                "name": f"{name_ar} ({ticker})",
                "percentage": weights[weight_idx],
            })
            weight_idx += 1

        chosen_names_for_text = [name for (_, name) in chosen]
        strategy_text = build_strategy_text(amount, risk, sector, chosen_names_for_text)

        return jsonify({
            "strategy": strategy_text,
            "portfolio": portfolio,
            "meta": {
                "risk": risk,
                "sector": sector,
                "amount": amount,
                "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
                "data_source": "yfinance (live)" if YFINANCE_AVAILABLE else "fallback (static order)",
            }
        }), 200

    except Exception as e:
        return jsonify({"error": f"حدث خطأ داخلي في الخادم: {str(e)}"}), 500


# =================================================================
# 5) نقطة فحص صحة الخادم (اختيارية - مفيدة عند الربط مع ngrok)
# =================================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "yfinance_available": YFINANCE_AVAILABLE,
        "sectors": list(SECTOR_POOLS.keys()),
    }), 200


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "المستثمر الذكي API",
        "endpoints": {
            "POST /predict": "توليد استراتيجية استثمار ومحفظة مقترحة",
            "GET /health": "فحص حالة الخادم",
        }
    }), 200


if __name__ == "__main__":
    # ملاحظة: في Colab استخدم pyngrok لفتح المنفذ للعالم الخارجي (راجع التعليمات أعلى الملف)
    app.run(host="0.0.0.0", port=5000, debug=True)
