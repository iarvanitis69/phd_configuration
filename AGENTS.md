1) Use the conda env interpreter phd_conda_env_p10 for this project
2) For the creation of README.pdf use the following command
   pandoc README.md -o README.pdf   --pdf-engine=xelatex   --toc --toc-depth=3   --number-sections   --highlight-style=tango   -V geometry:margin=0.8in   -V mainfont="DejaVu Serif"   -V sansfont="DejaVu Sans"   -V monofont="DejaVu Sans Mono"   -V colorlinks=true   -V linkcolor=blue   -V urlcolor=blue
3) after changes of code inform the README.md file