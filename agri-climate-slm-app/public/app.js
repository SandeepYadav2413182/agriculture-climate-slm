/* Shared frontend helpers: auth state, nav, API calls. */
const TOKEN_KEY = "acs_token";
const USER_KEY = "acs_user";

function getToken() { return localStorage.getItem(TOKEN_KEY); }
function getUser() { return localStorage.getItem(USER_KEY); }
function setSession(token, username) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, username);
}
function clearSession() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

async function api(path, opts = {}) {
  const headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
  const token = getToken();
  let body = opts.body;
  if (token) {
    if (body) {
      const parsed = JSON.parse(body);
      if (!parsed.token) parsed.token = token;
      body = JSON.stringify(parsed);
    } else {
      // GET requests carry no body: send the token as a Bearer header
      // (the server also accepts ?token= and body.token).
      headers["Authorization"] = "Bearer " + token;
    }
  }
  const res = await fetch(path, {
    method: opts.method || (body ? "POST" : "GET"),
    headers,
    body: body || undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function renderAuthArea() {
  const el = document.getElementById("authArea");
  if (!el) return;
  const user = getUser();
  if (user && getToken()) {
    el.innerHTML = `<span>Hi, <b>${esc(user)}</b></span><button class="btn small ghost" id="logoutBtn">Logout</button>`;
    document.getElementById("logoutBtn").addEventListener("click", async () => {
      try { await api("/api/logout", { body: JSON.stringify({}) }); } catch (_) {}
      clearSession();
      window.location.href = "index.html";
    });
  } else {
    el.innerHTML = `<a href="login.html">Login</a><a href="register.html" class="btn small">Register</a>`;
  }
}

function markNav() {
  const page = document.body.dataset.page;
  document.querySelectorAll("[data-nav]").forEach((a) => {
    if (a.dataset.nav === page) a.classList.add("active");
  });
}

function requireAuth() {
  if (!getToken()) {
    const next = encodeURIComponent(window.location.pathname.split("/").pop() || "advisor.html");
    window.location.href = `login.html?next=${next}`;
    return false;
  }
  return true;
}

/* Fluid page transitions: animate content in on load, glide out before
   following a sidebar/nav link instead of a hard static jump. */
function pageIn() {
  document.body.classList.add("page-in");
}

document.addEventListener("click", (e) => {
  const a = e.target.closest("a[href]");
  if (!a) return;
  let url;
  try { url = new URL(a.getAttribute("href"), window.location.href); }
  catch (_) { return; }
  const sameDoc = url.origin === window.location.origin;
  const _p = url.pathname.toLowerCase();
  const isPage = _p.endsWith(".html") || _p === "/" || _p.endsWith("/");
  const samePage = url.pathname === window.location.pathname && !url.hash;
  if (!sameDoc || !isPage || samePage || e.metaKey || e.ctrlKey || e.shiftKey || e.button === 1) return;
  e.preventDefault();
  document.body.classList.add("leaving");
  window.setTimeout(() => { window.location.href = a.href; }, 300);
});

document.addEventListener("DOMContentLoaded", () => {
  renderAuthArea();
  markNav();
  pageIn();
});
