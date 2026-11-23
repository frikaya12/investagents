let lastRef = null;
document.querySelectorAll("a.gl").forEach(a => a.addEventListener("click", () => { lastRef = a.id; }));
document.querySelectorAll("a.back").forEach(a => a.addEventListener("click", e => {
  if (lastRef && document.getElementById(lastRef).getAttribute("href") === "#" + a.dataset.term) {
    e.preventDefault();
    const el = document.getElementById(lastRef);
    el.scrollIntoView({behavior: "smooth", block: "center"});
    history.replaceState(null, "", "#" + lastRef);
    el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
  }
}));
const links = [...document.querySelectorAll("nav.toc a[href^='#s-']")];
const obs = new IntersectionObserver(es => es.forEach(en => {
  if (en.isIntersecting) { links.forEach(l => l.classList.toggle("active",
    l.getAttribute("href") === "#" + en.target.id)); }
}), {rootMargin: "0px 0px -75% 0px"});
document.querySelectorAll("main h2[id]").forEach(h => obs.observe(h));
