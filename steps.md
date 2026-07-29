# Build website (local jemdoc.py works with Python 3; no conda py2 env needed)
python jemdoc.py *.jemdoc

# Deploy to GitHub
% git add .  
% git commit -m "add ..."
% git push -u origin main 

% https://bingtan.me/jemdocuse.html