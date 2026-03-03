# Chengqian Using His Own Model Guide
## Connor Jason

This is a guide on how you can run this application within docker and integrate your own models using the model integration guide(/SharkMQP26/backend/custom_model_integration.md) for your own local development or showcasing purposes.

---

## Prerequisites

[Docker Desktop](https://www.docker.com/products/docker-desktop/)

After installing, open Docker Desktop and wait for the whale icon in your menu bar to say "Docker Desktop is running". Then in Settings -> Resources and ensure memory has at least like 4-6 GB.

---

## 1. Clone the Code

If you're here you proabbly already did that but:

```bash
git clone https://github.com/amgiovenco/SharkMQP26.git
cd SharkMQP26
```

---

## 2. First-Time Setup

Copy the env variables

```bash
cp .env.docker .env
```

and build the containers

```bash
docker compose up --build
```

The first build took me like 2 minutes ish but it'll be quicker after that.

When you stop seeing the blue docker building text and it changes to colorful logs from nginx, redis, postgres, etc, open **http://localhost** in your browser.

---

## 3. Log In

The database is auto-populated with the following accounts:

| Email | Password |
|-------|----------|
| `cejason@wpi.edu` | `wpiwpiwpi` |
| `amgiovenco@wpi.edu` | `wpiwpiwpi` |
| `kmlee@wpi.edu` | `wpiwpiwpi` |
| `czhang12@wpi.edu` | `wpiwpiwpi` |

---

## 4. Normal Commands

After you build the docker containers, you can use these commands to start/stop them quickly:

```bash
# Start
docker compose up

# Start in background
docker compose up -d

# Stop
docker compose down

# Wipe the database and start fresh
docker compose down -v && docker compose up

# Wipe everything and start fresh
docker compose down -v && docker system prune -f && docker compose up --build
```

---

## 5. Adding Your Own Model

Docker builds the container from your local files so changes you make on your Mac get picked up when you rebuild.

### Step 1 — Write your inference script

All the instructions are the same as before for adding a custom model. Please refer to the other file if you need a reference here: backend/custom_model_integration.md.

### Step 2 — Rebuild the worker

Since you changed Python code in "worker", the container needs to be rebuilt:

```bash
docker compose up --build worker
```

---

## 6. Saving Your Work

If you want to save your in-code changes, you can use normal git stuff. You can make your own branch in our repository, that's fine. Branch off of main and occasionally merge in the new changes if we make updates.

The database lives in a Docker volume (`sharkid-db`). It survives `docker compose down` but is wiped by `docker compose down -v`

---

## 7. Viewing Logs

```bash
# All services
docker compose logs -f

# Just the ML worker
docker compose logs -f worker

# Just the API
docker compose logs -f api
```