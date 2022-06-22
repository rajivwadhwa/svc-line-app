mkdir -p ~/.streamlit/

echo "\
"[theme]\n\
font = ‘serif’
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
