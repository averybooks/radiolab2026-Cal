# radiolab2026-Cal
Code for projects in UC Berkeley UG Astronomy Radio Labratory.

If for some reason there is a failure of git pull and you get a "Merge Conflict" error then run
  "nbdime merge-web" in terminal
this will open a web page showing your version on left and other version on right and you can select which to save.

Infrastructure for Git:
Inside bash: (ideally use Git Bash or Anaconda Prompt)
  1. pip install nbdime nbstripout
  2. nbdime config-git --enable --global
  3. nbstripout --install --global
  4. git clone https://github.com/USERNAME/radiolab2026-Cal.git
  5. git checkout YOUR BRANCH NAME
  6. git branch -> tells you which branch you're in (please only work within your branch, pushes to main from pi will be done as a group)

Working with Git:
  1. git branch -> double check you're in your own branch
  2. git pull
  3. edit code and add files 
  4. git add . -> adds all changes to your push
  5. git commit -m "update info"
  6. git push

specific Instructions for Git push and pull:
  1. "git add (notebookname).ipynb" will add changes -> think of it like putting items into a box before mailing it. You can select which files to be part of this particular update (say 3 files out of 10 in the github).
  2. "git commit -m "Updated analysis"" takes everything currently in the box from the add function and seals it permanently asa  snapshot in the local history. the part following the "-m" is a message for the update so you know what you did with this particular box. THIS WILL CLEAN THE NOTEBOOK so when you commit the NBSTRIPOUT tool makes sure the version getting saved has not graphs or output, just pure code
  3. "git push" sends updated files and code to the github
  4. "git pull" pulls the most recent files from the github


