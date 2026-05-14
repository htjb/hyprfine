window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],      // added $...$
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],   // added $$...$$
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};