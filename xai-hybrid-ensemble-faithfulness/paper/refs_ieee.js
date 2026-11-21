// IEEE-abbreviated reference list for the conference paper. Base texts from synopsis/refs.js (checked earlier),
// new entries verified online on 21 Nov 2026.
const R = require("./refs_base.js");
const ABBR = [["Nature Medicine", "Nat. Med."], ["Machine Learning,", "Mach. Learn.,"], ["Scientific Reports", "Sci. Rep."],
  ["Frontiers in Pharmacology", "Front. Pharmacol."], ["ACM Computing Surveys", "ACM Comput. Surv."], ["Healthcare,", "Healthcare,"],
  ["Diagnostics,", "Diagnostics,"], ["PLOS ONE", "PLoS One"], ["Big Data and Cognitive Computing", "Big Data Cogn. Comput."],
  ["Artificial Intelligence, vol.", "Artif. Intell., vol."], ["Statistics and Computing", "Stat. Comput."],
  ["Journal of Machine Learning Research", "J. Mach. Learn. Res."], ["Neural Networks", "Neural Netw."],
  ["Scandinavian Journal of Statistics", "Scand. J. Statist."], ["Controlled Clinical Trials", "Control. Clin. Trials"],
  ["Statistics in Medicine", "Stat. Med."], ["The American Journal of Cardiology", "Am. J. Cardiol."], ["Knowledge-Based Systems", "Knowl.-Based Syst."], ["IEEE Transactions on Biomedical Engineering", "IEEE Trans. Biomed. Eng."], ["Advances in Neural Information Processing Systems", "Adv. Neural Inf. Process. Syst."],
  ["Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining", "Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min."],
  ["Proc. 29th Int. Joint Conf. Artificial Intelligence (IJCAI)", "Proc. 29th Int. Joint Conf. Artif. Intell. (IJCAI)"],
  ["Transactions on Machine Learning Research, 2024.", "Trans. Mach. Learn. Res., 2024. [Online]. Available: arXiv:2202.01602"]];
const fix = t => ABBR.reduce((s, [a, b]) => s.split(a).join(b), t);
const NEW = {
  brfss: "Centers for Disease Control and Prevention, “Behavioral Risk Factor Surveillance System Survey Data, 2015.” Atlanta, GA, USA: U.S. Dept. of Health and Human Services. [Online]. Available: https://www.cdc.gov/brfss/annual_data/annual_2015.html",
  uci_cdc: "“CDC Diabetes Health Indicators,” UCI Machine Learning Repository, dataset ID 891. [Online]. Available: https://archive.ics.uci.edu/dataset/891 (accessed Nov. 2026).",
  divya: "K. S. Divya, G. F. Hussain, and Mohebbanaaz, “Explainable stacked ensemble model for robust and reliable heart disease diagnosis,” in Proc. IEEE Madhya Pradesh Section Conf. (MPCON), 2026, doi: 10.1109/MPCON69668.2026.11508581.",
  janzing: "D. Janzing, L. Minorics, and P. Blöbaum, “Feature relevance quantification in explainable AI: A causal problem,” in Proc. 23rd Int. Conf. Artif. Intell. Statist. (AISTATS), PMLR vol. 108, 2020.",
  petsiuk: "V. Petsiuk, A. Das, and K. Saenko, “RISE: Randomized input sampling for explanation of black-box models,” in Proc. British Mach. Vis. Conf. (BMVC), 2018.",
  rong: "Y. Rong, T. Leemann, V. Borisov, G. Kasneci, and E. Kasneci, “A consistent and efficient evaluation strategy for attribution methods,” in Proc. 39th Int. Conf. Mach. Learn. (ICML), PMLR vol. 162, 2022.",
  chen: "H. Chen, J. D. Janizek, S. Lundberg, and S.-I. Lee, “True to the model or true to the data?” arXiv:2006.16234, 2020.",
  covert: "I. Covert, S. Lundberg, and S.-I. Lee, “Explaining by removing: A unified framework for model explanation,” J. Mach. Learn. Res., vol. 22, no. 209, pp. 1–90, 2021.",
  sundararajan: "M. Sundararajan and A. Najmi, “The many Shapley values for model explanation,” in Proc. 37th Int. Conf. Mach. Learn. (ICML), PMLR vol. 119, 2020.",
  kumar: "I. E. Kumar, S. Venkatasubramanian, C. Scheidegger, and S. Friedler, “Problems with Shapley-value-based explanations as feature importance measures,” in Proc. 37th Int. Conf. Mach. Learn. (ICML), PMLR vol. 119, 2020.",
  inthout: "J. IntHout, J. P. A. Ioannidis, and G. F. Borm, “The Hartung-Knapp-Sidik-Jonkman method for random effects meta-analysis is straightforward and considerably outperforms the standard DerSimonian-Laird method,” BMC Med. Res. Methodol., vol. 14, Art. no. 25, 2014, doi: 10.1186/1471-2288-14-25.",
  riley: "R. D. Riley, J. P. T. Higgins, and J. J. Deeks, “Interpretation of random effects meta-analyses,” BMJ, vol. 342, Art. no. d549, 2011, doi: 10.1136/bmj.d549.",
  uci_repo: "M. Kelly, R. Longjohn, and K. Nottingham, “The UCI Machine Learning Repository.” [Online]. Available: https://archive.ics.uci.edu (accessed Nov. 2026).",
  teboul: "A. Teboul, “Diabetes Health Indicators Dataset” (from CDC BRFSS 2015), Kaggle, 2021. [Online]. Available: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset",
};
module.exports = k => NEW[k] || (R[k] ? fix(R[k].t) : (() => { throw new Error("ref " + k); })());
