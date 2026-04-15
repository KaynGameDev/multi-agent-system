const elements = {
  loginForm: document.getElementById("loginForm"),
  usernameInput: document.getElementById("usernameInput"),
  passwordInput: document.getElementById("passwordInput"),
  loginButton: document.getElementById("loginButton"),
  loginError: document.getElementById("loginError"),
};

async function login(username, password) {
  const response = await fetch("/api/auth/login", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      username,
      password,
    }),
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || "Sign in failed.");
  }

  return response.json();
}

elements.loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  elements.loginError.hidden = true;
  elements.loginButton.disabled = true;

  try {
    await login(
      elements.usernameInput.value.trim(),
      elements.passwordInput.value,
    );
    window.location.href = "/";
  } catch (error) {
    elements.loginError.textContent = error.message || "Sign in failed.";
    elements.loginError.hidden = false;
    elements.passwordInput.select();
  } finally {
    elements.loginButton.disabled = false;
  }
});
