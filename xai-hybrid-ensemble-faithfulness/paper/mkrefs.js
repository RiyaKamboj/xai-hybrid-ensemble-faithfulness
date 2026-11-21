const fs=require("fs");const ref=require("./refs_ieee.js");
const order=[];for(const f of ["sec_abstract.tex","sec_front.tex","sec_methods.tex","main.tex","sec_results.tex","sec_discussion.tex"]){const t=fs.readFileSync(f,"utf8");for(const m of t.matchAll(/\\cite\{([^}]+)\}/g))for(const k of m[1].split(","))if(!order.includes(k.trim()))order.push(k.trim());}
const esc=s=>s.replace(/&/g,"\\&").replace(/%/g,"\\%").replace(/_/g,"\\_").replace(/#/g,"\\#").replace(/“/g,"``").replace(/”/g,"''").replace(/‘/g,"`").replace(/’/g,"'").replace(/–/g,"--").replace(/—/g,"---").replace(/ø/g,"{\\o}").replace(/ö/g,"\\\"{o}").replace(/š/g,"\\v{s}");
const urlify=s=>s.replace(/doi: (\S+?)\.?$/,(m,d)=>"doi: \\url{"+d.replace(/\\_/g,"_")+"}.").replace(/Available: (https?:\S+)/,(m,u)=>"Available: \\url{"+u.replace(/\\_/g,"_")+"}");
fs.writeFileSync("refs.tex",order.map(k=>`\\bibitem{${k}} ${urlify(esc(ref(k)))}`).join("\n"));
console.log(order.length,"refs");
