/* Solarpunk farm scene: golden-hour fields, sun, sunflowers, greenhouse,
   butterflies and drifting pollen. Pure agriculture, no city.
   Inline SVG + CSS keyframes, subtle mouse parallax on depth layers. */
(function () {
  var seed = 20261005;
  function rnd() { seed = (seed * 16807) % 2147483647; return seed / 2147483647; }

  var S = [];

  /* ---------- defs ---------- */
  S.push('<defs>'
    + '<linearGradient id="fmSky" x1="0" y1="0" x2="0" y2="1">'
    + '<stop offset="0%" stop-color="#274b5e"/>'
    + '<stop offset="48%" stop-color="#7a7a4a"/>'
    + '<stop offset="72%" stop-color="#d99a4e"/>'
    + '<stop offset="100%" stop-color="#f2b96a"/>'
    + '</linearGradient>'
    + '<radialGradient id="fmSunGlow" cx="50%" cy="50%" r="50%">'
    + '<stop offset="0%" stop-color="#ffe9a8" stop-opacity="0.85"/>'
    + '<stop offset="45%" stop-color="#ffd97a" stop-opacity="0.35"/>'
    + '<stop offset="100%" stop-color="#ffd97a" stop-opacity="0"/>'
    + '</radialGradient>'
    + '<radialGradient id="fmVig" cx="50%" cy="46%" r="75%">'
    + '<stop offset="62%" stop-color="#000" stop-opacity="0"/>'
    + '<stop offset="100%" stop-color="#0a0803" stop-opacity="0.5"/>'
    + '</radialGradient>'
    + '</defs>');

  /* ---------- sky ---------- */
  S.push('<rect x="0" y="0" width="1600" height="900" fill="url(#fmSky)"/>');

  /* ---------- sun (parallax layer 1) ---------- */
  S.push('<g class="px" data-depth="2"><g transform="translate(1060,330)">'
    + '<g class="sun-breathe"><circle r="230" fill="url(#fmSunGlow)"/></g>'
    + '<circle r="72" fill="#ffe6a0"/>'
    + '<circle r="72" fill="none" stroke="rgba(255,240,200,0.6)" stroke-width="2"/>'
    + '<circle class="spin-slow" r="112" fill="none" stroke="rgba(255,220,140,0.6)"'
    + ' stroke-width="8" stroke-dasharray="16 34" stroke-linecap="round"/>'
    + '</g></g>');

  /* ---------- clouds (parallax layer 2) ---------- */
  function cloud(x, y, s, cls, dur) {
    return '<g class="cloud ' + cls + '" style="animation-duration:' + dur + 's">'
      + '<g transform="translate(' + x + ',' + y + ') scale(' + s + ')">'
      + '<ellipse cx="0" cy="0" rx="80" ry="26" fill="rgba(255,244,220,0.75)"/>'
      + '<ellipse cx="-48" cy="9" rx="48" ry="18" fill="rgba(255,244,220,0.7)"/>'
      + '<ellipse cx="50" cy="10" rx="52" ry="19" fill="rgba(255,244,220,0.7)"/>'
      + '<ellipse cx="0" cy="-10" rx="62" ry="19" fill="rgba(255,250,235,0.8)"/>'
      + '</g></g>';
  }
  S.push('<g class="px" data-depth="5">'
    + cloud(330, 170, 1.1, "c1", 110)
    + cloud(820, 120, 0.75, "c2", 150)
    + cloud(1380, 210, 0.9, "c3", 130)
    + '<g class="cloud c2" style="animation-duration:120s"><g transform="translate(620,260)">'
    + '<path d="M0,0 q8,-8 16,0 q8,-8 16,0" stroke="rgba(60,50,40,0.55)" stroke-width="3" fill="none" stroke-linecap="round"/>'
    + '<path d="M40,18 q7,-7 14,0 q7,-7 14,0" stroke="rgba(60,50,40,0.45)" stroke-width="2.5" fill="none" stroke-linecap="round"/>'
    + '</g></g>'
    + '</g>');

  /* ---------- rolling hills + tilled fields (parallax layer 3) ---------- */
  var field = '<g class="px" data-depth="8">';
  field += '<g class="layer-sway" style="animation-duration:55s">'
    + '<path d="M-40,620 C260,540 520,600 820,580 C1120,560 1360,600 1640,570 L1640,900 L-40,900 Z" fill="#274d2c"/>'
    + '<path d="M-40,700 C300,640 640,690 980,670 C1260,655 1460,690 1640,670 L1640,900 L-40,900 Z" fill="#1d3d22"/>'
    + '<path d="M-40,790 C340,740 700,780 1080,765 C1330,756 1500,780 1640,770 L1640,900 L-40,900 Z" fill="#16301a"/>'
    + '</g>';
  /* perspective crop rows */
  field += '<g stroke="rgba(255,235,180,0.16)" stroke-width="3">';
  for (var r = 0; r < 11; r++) {
    var bx = -200 + r * 200;
    field += '<line x1="800" y1="640" x2="' + bx + '" y2="900"/>';
  }
  field += '</g>';
  /* greenhouse dome */
  field += '<g opacity="0.95" transform="translate(1240,668)">'
    + '<path d="M-70,0 A70,70 0 0 1 70,0 Z" fill="rgba(220,240,220,0.18)" stroke="rgba(240,250,235,0.5)" stroke-width="2"/>'
    + '<line x1="0" y1="-70" x2="0" y2="0" stroke="rgba(240,250,235,0.35)" stroke-width="1.5"/>'
    + '<line x1="-70" y1="0" x2="70" y2="0" stroke="rgba(240,250,235,0.5)" stroke-width="2"/>'
    + '<circle cx="-30" cy="-26" r="4" fill="#ffe9a0" class="led"/>'
    + '<circle cx="26" cy="-40" r="4" fill="#ffe9a0" class="led" style="animation-delay:-1.3s"/>'
    + '</g>';
  field += '</g>';
  S.push(field);

  /* ---------- sunflowers + crop sprouts (parallax layer 4) ---------- */
  function sunflower(x, baseY, s, delay) {
    var petals = "";
    for (var a = 0; a < 360; a += 36) {
      petals += '<ellipse cx="0" cy="-27" rx="9" ry="24" fill="#ffcf5e" transform="rotate(' + a + ')"/>';
    }
    return '<g transform="translate(' + x + ',' + baseY + ') scale(' + s + ')">'
      + '<g class="sway2" style="animation-delay:' + delay + 's">'
      + '<path d="M0,0 C-6,-42 6,-84 0,-124" stroke="#2f7d44" stroke-width="7" fill="none" stroke-linecap="round"/>'
      + '<path d="M0,-62 C-26,-68 -36,-88 -38,-104 C-22,-96 -8,-80 0,-62 Z" fill="#3a9a52"/>'
      + '<path d="M0,-86 C16,-92 26,-108 28,-122 C14,-114 4,-100 0,-86 Z" fill="#35904c"/>'
      + '<g transform="translate(0,-134)">' + petals
      + '<circle r="17" fill="#8a5a1e"/><circle r="11" fill="#5e3c12"/>'
      + '</g></g></g>';
  }
  var mid = '<g class="px" data-depth="12">';
  var sfPos = [[180, 760, 1.0], [330, 790, 1.25], [520, 745, 0.9], [700, 800, 1.15],
               [900, 750, 0.95], [1120, 795, 1.2], [1330, 755, 1.0], [1500, 790, 1.1]];
  for (var i = 0; i < sfPos.length; i++) {
    mid += sunflower(sfPos[i][0], sfPos[i][1], sfPos[i][2], (-rnd() * 6).toFixed(1));
  }
  /* sprout rows */
  mid += '<g stroke="#4caf6a" stroke-width="3" fill="none" stroke-linecap="round">';
  for (var sp = 0; sp < 22; sp++) {
    var sx = 30 + sp * 72 + rnd() * 24, sy = 845 + rnd() * 30;
    mid += '<g transform="translate(' + sx.toFixed(0) + ',' + sy.toFixed(0) + ')">'
      + '<g class="sway2" style="animation-delay:-' + (rnd() * 6).toFixed(1) + 's;animation-duration:4.5s">'
      + '<path d="M0,0 C-2,-12 -10,-18 -18,-20 M0,0 C2,-12 10,-18 18,-20 M0,0 L0,-14"/>'
      + '</g></g>';
  }
  mid += '</g></g>';
  S.push(mid);

  /* ---------- butterflies ---------- */
  function butterfly(x, y, dur, delay, color) {
    return '<g class="butterfly" style="animation-duration:' + dur + 's;animation-delay:' + delay + 's">'
      + '<g transform="translate(' + x + ',' + y + ')">'
      + '<ellipse cx="-6" cy="0" rx="7" ry="10" fill="' + color + '" opacity="0.85" transform="rotate(-24)"/>'
      + '<ellipse cx="6" cy="0" rx="7" ry="10" fill="' + color + '" opacity="0.85" transform="rotate(24)"/>'
      + '<rect x="-1.6" y="-9" width="3.2" height="18" rx="1.6" fill="#3a2c18"/>'
      + '</g></g>';
  }
  S.push(butterfly(500, 480, 22, -4, "#ffd97a"));
  S.push(butterfly(1050, 540, 28, -15, "#ffb3c8"));
  S.push(butterfly(220, 560, 25, -10, "#fff3d6"));

  /* ---------- floating pollen ---------- */
  for (var f = 0; f < 12; f++) {
    var fx = rnd() * 1600, fy = 380 + rnd() * 420;
    S.push('<circle class="mote" style="animation-delay:-' + (rnd() * 10).toFixed(1)
      + 's;animation-duration:' + (8 + rnd() * 6).toFixed(1) + 's"'
      + ' cx="' + fx.toFixed(0) + '" cy="' + fy.toFixed(0) + '" r="3" fill="#ffedb8"/>');
  }

  /* ---------- foreground wheat (parallax layer 5) ---------- */
  var fg = '<g class="px" data-depth="16"><g stroke="#c9a44e" stroke-width="4" fill="none" stroke-linecap="round">';
  for (var w = 0; w < 16; w++) {
    var wx = 20 + w * 102 + rnd() * 40, wh = 70 + rnd() * 60, bend = (rnd() - 0.5) * 70;
    fg += '<g transform="translate(' + wx.toFixed(0) + ',905)">'
      + '<g class="sway2" style="animation-delay:-' + (rnd() * 5).toFixed(1) + 's">'
      + '<path d="M0,0 C' + (bend * 0.3).toFixed(0) + ',' + (-wh * 0.6).toFixed(0) + ' '
      + bend.toFixed(0) + ',' + (-wh * 0.8).toFixed(0) + ' ' + (bend * 1.2).toFixed(0) + ',' + (-wh).toFixed(0) + '"/>'
      + '<ellipse cx="' + (bend * 1.2).toFixed(0) + '" cy="' + (-wh).toFixed(0) + '" rx="7" ry="16" fill="#e0b95e" stroke="none" opacity="0.9"/>'
      + '</g></g>';
  }
  fg += '</g></g>';
  S.push(fg);

  /* ---------- vignette ---------- */
  S.push('<rect x="0" y="0" width="1600" height="900" fill="url(#fmVig)"/>');

  var wrap = document.createElement("div");
  wrap.setAttribute("aria-hidden", "true");
  wrap.innerHTML = '<svg id="bg-farm" viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice">'
    + S.join("") + "</svg>";
  document.body.insertBefore(wrap, document.body.firstChild);

  /* ---------- subtle mouse parallax ---------- */
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var touch = window.matchMedia && window.matchMedia("(pointer: coarse)").matches;
  if (!reduce && !touch) {
    var layers = Array.prototype.slice.call(wrap.querySelectorAll(".px"));
    var tx = 0, ty = 0, cx = 0, cy = 0;
    window.addEventListener("mousemove", function (e) {
      tx = (e.clientX / window.innerWidth - 0.5);
      ty = (e.clientY / window.innerHeight - 0.5);
    }, { passive: true });
    (function tick() {
      cx += (tx - cx) * 0.06;
      cy += (ty - cy) * 0.06;
      for (var i = 0; i < layers.length; i++) {
        var d = parseFloat(layers[i].getAttribute("data-depth")) || 0;
        layers[i].style.transform = "translate(" + (-cx * d * 2.2).toFixed(1)
          + "px," + (-cy * d * 1.6).toFixed(1) + "px)";
      }
      requestAnimationFrame(tick);
    })();
  }
})();
