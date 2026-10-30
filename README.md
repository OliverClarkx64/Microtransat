"# Microtransat" 
---setup git in a directory--- (do this once)
setup user config (make github account):
git --version (to check git version)
git config --global user.name "your name here"
git config --global user.email "your email here"

then in directory you want the work to appear (windows)
copy your filepath from files and in CMD or Powershell
CD "filepath"
git init 
git clone https://github.com/OliverClarkx64/Microtransat
git remote add origin https://github.com/OliverClarkx64/Microtransat
git pull origin main

^^now you have the files setup on your system^^




---How to keep files up to date--- (do this every time you work)
clone/pull github before doing any work (in CMD or Powershell):
    -git pull origin master
Then make your edits however you want
Clone again before commiting anything
    -git pull origin main
Then add any additional files:
    -git add -A
Commit changes with small note of whats changed:
    -git commit -a -m "message for commit"
push update:
    -git push origin main

And You're done