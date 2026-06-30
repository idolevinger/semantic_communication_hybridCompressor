import markdown
import codecs
import re

with codecs.open('experiment_results.md', 'r', 'utf-8') as f:
    text = f.read()

# Remove the <style> block we injected earlier
text = re.sub(r'<style>.*?</style>', '', text, flags=re.DOTALL)

html_content = markdown.markdown(text, extensions=['fenced_code', 'tables'])

final_html = f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="utf-8">
<title>דוח ניסויים</title>
<style>
body {{
    direction: rtl;
    text-align: right;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 40px auto;
    max-width: 900px;
    line-height: 1.6;
    font-size: 16px;
    padding: 0 20px;
    color: #333;
}}
h1, h2, h3 {{
    color: #2c3e50;
    border-bottom: 1px solid #eee;
    padding-bottom: 10px;
}}
img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 20px auto;
    border: 1px solid #ddd;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}}
pre, code {{
    direction: ltr;
    text-align: left;
    background: #f4f4f4;
    padding: 2px 5px;
    border-radius: 4px;
    font-family: Consolas, monospace;
}}
pre {{
    padding: 15px;
    overflow-x: auto;
}}
blockquote {{
    border-right: 5px solid #3498db;
    margin: 0;
    padding: 10px 20px;
    background-color: #f8f9fa;
    color: #555;
}}
</style>
</head>
<body>
{html_content}
</body>
</html>"""

with codecs.open('results_report.html', 'w', 'utf-8') as f:
    f.write(final_html)

print("HTML report successfully created!")
