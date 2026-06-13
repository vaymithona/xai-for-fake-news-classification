"""
Three representative short articles for the gr.Examples component.
One confidently fake, one confidently real, one borderline/satirical.
Keep under ~120 words so SHAP runs quickly during live demos.
"""

# Each entry: [article_text]  (single-column examples, matching the text input)

FAKE_ARTICLE = """\
BREAKING: Scientists confirm that drinking bleach cures coronavirus, according \
to sources close to the White House. President's personal physician reportedly \
endorsed the treatment after secret trials showed 100% success rate. Mainstream \
media refuses to cover this bombshell story to protect Big Pharma profits. \
Share this before it gets deleted! Thousands of patriots already cured using \
this one simple trick that doctors don't want you to know. The deep state is \
working overtime to suppress the truth. God bless America and our brave president \
for standing up against the radical left globalist agenda.\
"""

REAL_ARTICLE = """\
Federal Reserve officials held interest rates steady on Wednesday and signaled \
they are in no rush to cut borrowing costs, citing continued strength in the \
labor market and persistent inflation above the central bank's two-percent target. \
Chair Jerome Powell said at a press conference that policymakers want to see \
more evidence that inflation is sustainably moving toward the target before \
reducing rates. The decision was unanimous among the twelve-member Federal Open \
Market Committee. Futures markets lowered expectations for a rate cut at the \
next meeting following the announcement.\
"""

BORDERLINE_ARTICLE = """\
A widely shared image claiming to show a leaked government memo ordering \
hospitals to classify all deaths as COVID-19 to inflate pandemic statistics \
has been circulating on social media. Fact-checkers at Reuters and Associated \
Press found the document was fabricated and not consistent with official \
government letterhead. Hospital administrators contacted by journalists denied \
any such directive exists. The claim stems from a network of websites that \
have previously published false health information. Social media platforms \
removed some versions of the post under misinformation policies.\
"""

EXAMPLES = [
    [FAKE_ARTICLE],
    [REAL_ARTICLE],
    [BORDERLINE_ARTICLE],
]

EXAMPLE_LABELS = ['Confident Fake', 'Confident Real', 'Borderline / Fact-check']
