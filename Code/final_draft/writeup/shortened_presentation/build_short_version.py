"""Reorder existing seminar frames without changing their text or figures."""
from pathlib import Path
import re
import hashlib
import json

source = Path('/Users/tate/LOV_CC_2YP/Code/final_draft/seminar_pres/pres.tex')
out = Path(__file__).resolve().parent
text = source.read_text()
frames = re.findall(r'\\begin\{frame\}.*?\\end\{frame\}', text, re.S)
assert len(frames) == 63
main_groups = [
 ('Introduction', [1,2,3]),
 ('Motivation from Data', [4]),
 ('Supply Theory', [6,7,8,10,14,18,20]),
 ('Data', [21,24,25]),
 ('Demand Estimation', [26,27,29,32,31]),
 ('Pricing Exercise', [34,35,40,41,42,43,44,45,47,49]),
 ('Conclusion', [50,51]),
]
main = [n for _, numbers in main_groups for n in numbers]
assert len(main) == 31
# Each retained original frame appears exactly once, including all original backups.
appendix_groups = [
 ([22,23],21),
 ([5],4),
 ([9,11,52],10),
 ([12,13,15,16,17],14),
 ([19,53,54],20),
 ([28,55,56,61],29),
 ([30,33],32),
 ([62],31),
 ([36,37],44),
 ([38,39],42),
 ([46,57,58],47),
 ([48,59,60],49),
]
appendix = [n for numbers,_ in appendix_groups for n in numbers]
assert sorted(main+appendix) == list(range(1,63))
links = {
 21:[(22,'Activity'),(23,'Demographics')],
 4:[(5,'Flavor Popularity')],
 10:[(9,'Terminal FOC'),(11,'Comparative statics'),(52,'Elasticities')],
 14:[(12,'Calibration'),(13,'Table'),(15,'Profit'),(16,'Price ratio'),(17,'Discussion')],
 20:[(19,'Profit levels'),(53,'Price paths'),(54,'Price paths')],
 29:[(28,'Probabilities'),(55,'Price controls'),(56,'Controlled estimates'),(61,'First stages')],
 32:[(30,'Estimates'),(33,'Discussion')],
 31:[(62,'Controlled WTP')],
 44:[(36,'Berry types'),(37,'Plain types')],
 42:[(38,'Berry offers'),(39,'Plain offers')],
 47:[(46,'Revenue levels'),(57,'Price paths'),(58,'Price paths')],
 49:[(48,'Revenue levels'),(59,'Price paths'),(60,'Price paths')],
}
preamble = text.split(r'\begin{document}',1)[0]
# Navigation is placed in a separate footline row, outside unchanged frame bodies.
preamble += r'''
% Short-version navigation only. Original frame content is preserved verbatim.
\newcommand{\shortbuttons}{}
\newcommand{\shortmaincount}{31}
\newif\ifshortappendix
\shortappendixfalse
\setbeamertemplate{footline}{%
  \leavevmode\vbox{\hbox{%
  \begin{beamercolorbox}[wd=\paperwidth,ht=3.8ex,dp=.6ex,rightskip=1em]{normal text}%
    \hfill{\scriptsize\shortbuttons}
  \end{beamercolorbox}}%
  \hbox{\begin{beamercolorbox}[wd=\paperwidth,ht=2.5ex,dp=1ex,leftskip=1em,rightskip=1em]{footline}%
    \insertshortauthor{}\hfill
    \ifshortappendix Appendix\else\insertframenumber/\shortmaincount\fi
  \end{beamercolorbox}}}%
}
'''

approved_intro = {
1: r"""\begin{frame}{Motivation}
  \begin{itemize}
    \item Firms have more access than ever to consumer information
    \item Allows for targeted discounting based on preferences derived from repeat interactions
    \item Targeted discounts allow firms to steer consumers toward particular products.
  \end{itemize}
  \begin{tcolorbox}
    How do firms price based on repeated interactions with consumers and how are profits affected?
  \end{tcolorbox}
\end{frame}""",
2: r"""\begin{frame}{Research Question}
  \begin{tcolorbox}
    How can firms price discriminate via consumer preferences for novelty and habit and how much is information about these preferences worth?
  \end{tcolorbox}
  How do I answer?
  \begin{itemize}
    \item Solve a theoretical model to understand firm's behavior
    \item Estimate consumer's preferences with Nielsen data, recovering the distribution of novelty/habit preferences
    \item Use the estimated distribution in the pricing model to understand how firm's revenues are impacted by this sort of targeting
  \end{itemize}
\end{frame}""",
}

def title(n):
    match=re.match(r'\\begin\{frame\}(?:\[[^\]]*\])?\{([^}]*)\}',frames[n])
    return match.group(1) if match else 'Title'
def frame(n,backup=False):
    original=approved_intro.get(n,frames[n])
    if n == 3:
        # TeX escaping only: these existing author names contain literal ampersands.
        original=original.replace('Anand & Shachar',r'Anand \& Shachar').replace('Karle & Reisinger',r'Karle \& Reisinger')
    match=re.match(r'\\begin\{frame\}(?:\[([^\]]*)\])?',original)
    options=[v for v in (match.group(1) or '').split(',') if v]
    options.append('label=short-'+str(n))
    if backup and 'noframenumbering' not in options:options.append('noframenumbering')
    result=r'\begin{frame}['+','.join(options)+']'+original[match.end():]
    return result

def emit(n,backup=False,parent=None):
    if backup:
        destinations=[(parent,'Back to '+title(parent))]
    else:
        destinations=links.get(n,[])
    buttons=r'\hspace{.35em}'.join(r'\hyperlink{short-'+str(target)+r'}{\beamergotobutton{'+label+'}}' for target,label in destinations)
    content='\\gdef\\shortbuttons{'+buttons+'}\n'+frame(n,backup)+'\n'
    # Preserve the original conditional inclusion of these optional figures.
    if n in [40,41,43]:
        asset=re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}',frames[n]).group(1)
        content='\\IfFileExists{'+asset+'}{\n'+content+'}{}\n'
    return '% Original numbered slide '+str(n)+'\n'+content

parts=[preamble,r'\begin{document}',emit(0)]
for section,numbers in main_groups:
    parts.append('\\section{'+section+'}')
    parts.extend(emit(n) for n in numbers)
parts += [r'\appendix',r'\shortappendixtrue',r'\section{Appendix}']
for numbers,parent in appendix_groups:
    parts.extend(emit(n,True,parent) for n in numbers)
parts.append(r'\end{document}')
result='\n\n'.join(parts)+'\n'
(out/'pres_short.tex').write_text(result)
# Verify every original frame's body and title remain byte-for-byte identical;
# only the frame options gain labels and backup numbering flags.
newframes=re.findall(r'\\begin\{frame\}.*?\\end\{frame\}',result,re.S)
strip_options=lambda f: re.sub(r'^(\\begin\{frame\})(?:\[[^\]]*\])?',r'\1',f,count=1)
assert len(newframes)==len(frames)
order=[0]+main+appendix
for original_id,new in zip(order,newframes):
    assert strip_options(new).replace(r'\&','&')==strip_options(approved_intro.get(original_id,frames[original_id])).replace(r'\&','&'),original_id
# Validate that every button destination exists, including every appendix return.
labels=set(re.findall(r'label=(short-\d+)',result))
targets=set(re.findall(r'\\hyperlink\{(short-\d+)\}',result))
assert targets<=labels,targets-labels
report=dict(main_slides=len(main),appendix_slides=len(appendix),title_slides=1,
            original_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            main_original_ids=main,appendix_original_ids=appendix,
            unchanged_except_approved_introduction=True, tex_only_ampersand_escaping=True,button_targets=len(targets))
(out/'reorganization_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
