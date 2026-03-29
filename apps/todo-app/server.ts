import express from "express";
import path from "path";

const app = express();
const PORT = 3000;

app.use(express.static(path.join(__dirname, "..", "public")));
app.use(express.json());

interface Todo {
  id: string;
  title: string;
  completed: boolean;
  createdAt: string;
}

let todos: Todo[] = [];

app.get("/api/todos", (_req, res) => {
  res.json(todos);
});

app.post("/api/todos", (req, res) => {
  const title = req.body.title?.trim();
  if (!title) {
    res.status(400).json({ error: "Title is required" });
    return;
  }
  const todo: Todo = {
    id: Date.now().toString(),
    title,
    completed: false,
    createdAt: new Date().toISOString(),
  };
  todos.push(todo);
  res.status(201).json(todo);
});

app.patch("/api/todos/:id", (req, res) => {
  const todo = todos.find((t) => t.id === req.params.id);
  if (!todo) {
    res.status(404).json({ error: "Not found" });
    return;
  }
  Object.assign(todo, req.body);
  res.json(todo);
});

app.delete("/api/todos/:id", (req, res) => {
  todos = todos.filter((t) => t.id !== req.params.id);
  res.status(204).send();
});

// Reset endpoint — crucial for environment resets
app.post("/api/reset", (req, res) => {
  todos = req.body.todos || [];
  res.json({ status: "reset", count: todos.length });
});

app.get("/api/health", (_req, res) => {
  res.json({ status: "ok", todoCount: todos.length });
});

app.listen(PORT, () => {
  console.log(`Todo app running on port ${PORT}`);
});
