const form = document.getElementById("todo-form");
const input = document.getElementById("todo-input");
const list = document.getElementById("todo-list");
const stats = document.getElementById("stats");

async function loadTodos() {
  const res = await fetch("/api/todos");
  const todos = await res.json();
  renderTodos(todos);
}

function renderTodos(todos) {
  list.innerHTML = "";
  todos.forEach((todo) => {
    const li = document.createElement("li");
    li.dataset.testid = "todo-item-" + todo.id;
    li.className = todo.completed ? "completed" : "";
    li.innerHTML =
      '<input type="checkbox" data-testid="toggle-' + todo.id + '"' +
      (todo.completed ? " checked" : "") +
      ' onchange="toggleTodo(\'' + todo.id + "', this.checked)\" />" +
      '<span data-testid="todo-text-' + todo.id + '">' + todo.title + "</span>" +
      '<button data-testid="delete-' + todo.id + '"' +
      " onclick=\"deleteTodo('" + todo.id + "')\">×</button>";
    list.appendChild(li);
  });
  const total = todos.length;
  const done = todos.filter((t) => t.completed).length;
  stats.textContent = total > 0 ? done + "/" + total + " completed" : "";
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const title = input.value.trim();
  if (!title) return;
  await fetch("/api/todos", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  input.value = "";
  loadTodos();
});

window.toggleTodo = async (id, completed) => {
  await fetch("/api/todos/" + id, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ completed }),
  });
  loadTodos();
};

window.deleteTodo = async (id) => {
  await fetch("/api/todos/" + id, { method: "DELETE" });
  loadTodos();
};

loadTodos();
