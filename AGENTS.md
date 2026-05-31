1) Use the conda env interpreter phd_conda_env_p10 for this project
2) After changes on code 
   a) inform the README.md file
   b) create the README.pdf using the following command
   pandoc README.md -o README.pdf -V author="Arvanitis John"  --pdf-engine=xelatex   --toc --toc-depth=3   --number-sections   --highlight-style=tango   -V geometry:margin=0.8in   -V mainfont="DejaVu Serif"   -V sansfont="DejaVu Sans"   -V monofont="DejaVu Sans Mono"   -V colorlinks=true   -V linkcolor=blue   -V urlcolor=blue
