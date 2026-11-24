// SPDX-License-Identifier: AGPL-3.0-or-later
// The part of IndexedDB the chat drafts use, for jsdom, which has none: one
// database shared by every window it is installed in, readwrite transactions one
// after another, requests that answer later, and a transaction that completes
// once its requests are answered and none was added.

class Request {
  constructor() { this.result = undefined; this.error = null; }
}

const later = (fn) => setTimeout(fn, 0);
const named = (name, message) => Object.assign(new Error(message), { name });

export function makeIndexedDB() {
  const stores = new Map();   // store name -> Map of rows
  let version = 0;
  let writing = Promise.resolve();
  const fake = {
    rows: () => stores.get("rows") || new Map(),
    stores,
    connections: [],
    /** The next *n* readwrite transactions abort. */
    failWrites: 0,
    writes: 0,
  };

  class Transaction {
    constructor(db, name, mode) {
      this.db = db;
      this.mode = mode;
      this.pending = 0;
      this.undo = new Map(stores.get(name));
      this.name = name;
      this.work = [];
      this.state = "waiting";
      const start = () => {
        this.state = "active";
        if (mode === "readwrite" && fake.failWrites > 0) {
          fake.failWrites--;
          this.abortWith(named("QuotaExceededError", "no room"));
          return;
        }
        for (const w of this.work) w();
        this.check();
      };
      this.ended = new Promise((resolve) => { this.end = resolve; });
      if (mode === "readwrite") {
        const before = writing;
        writing = this.ended;
        before.then(() => later(start));
      } else later(start);
    }
    abortWith(e) {
      if (this.state === "done") return;
      this.state = "done";
      this.error = e;
      if (this.mode === "readwrite") stores.set(this.name, this.undo);
      if (this.onabort) this.onabort();
      this.end();
    }
    check() {
      later(() => {
        if (this.state !== "active" || this.pending) return;
        this.state = "done";
        if (this.mode === "readwrite") fake.writes++;
        if (this.oncomplete) this.oncomplete();
        this.end();
      });
    }
    request(op) {
      const req = new Request();
      this.pending++;
      const run = () => later(() => {
        if (this.state !== "active") return;
        req.result = op();
        this.pending--;
        try { if (req.onsuccess) req.onsuccess({ target: req }); } catch (e) { this.abortWith(e); return; }
        this.check();
      });
      if (this.state === "active") run();
      else this.work.push(run);
      return req;
    }
    objectStore() {
      const rows = () => stores.get(this.name);
      const copy = (v) => (v === undefined ? undefined : structuredClone(JSON.parse(JSON.stringify(v))));
      return {
        get: (key) => this.request(() => copy(rows().get(key))),
        put: (value, key) => { const v = copy(value); return this.request(() => { rows().set(key, v); return key; }); },
        delete: (key) => this.request(() => { rows().delete(key); }),
        openCursor: () => {
          const keys = [...rows().keys()];
          let i = -1;
          const req = new Request();
          const step = () => {
            this.pending++;
            const run = () => later(() => {
              if (this.state !== "active") return;
              i++;
              this.pending--;
              if (i >= keys.length) req.result = null;
              else {
                const key = keys[i];
                req.result = { key, value: copy(rows().get(key)), continue: step,
                               delete: () => this.request(() => { rows().delete(key); }) };
              }
              req.onsuccess({ target: req });
              this.check();
            });
            if (this.state === "active") run();
            else this.work.push(run);
          };
          step();
          return req;
        },
      };
    }
  }

  class Connection {
    constructor() { this.closed = false; fake.connections.push(this); }
    get objectStoreNames() { return [...stores.keys()]; }
    createObjectStore(name) { stores.set(name, new Map()); }
    deleteObjectStore(name) { stores.delete(name); }
    close() { this.closed = true; }
    transaction(name, mode) {
      if (this.closed) throw named("InvalidStateError", "The database connection is closing.");
      return new Transaction(this, name, mode);
    }
  }

  fake.api = {
    open(name, wanted) {
      const req = new Request();
      later(() => {
        const db = new Connection();
        req.result = db;
        if (wanted > version) {
          version = wanted;
          if (req.onupgradeneeded) req.onupgradeneeded({ target: req });
        }
        req.onsuccess({ target: req });
      });
      return req;
    },
  };
  return fake;
}

/** A BroadcastChannel class whose instances in different windows hear each other. */
export function makeChannels() {
  const open = new Set();
  return class {
    constructor(name) { this.name = name; this.listeners = new Set(); this.onmessage = null; open.add(this); }
    addEventListener(_type, fn) { this.listeners.add(fn); }
    removeEventListener(_type, fn) { this.listeners.delete(fn); }
    close() { open.delete(this); }
    postMessage(data) {
      const copy = JSON.parse(JSON.stringify(data));
      for (const other of open) {
        if (other === this || other.name !== this.name) continue;
        later(() => {
          if (other.onmessage) other.onmessage({ data: copy });
          for (const fn of other.listeners) fn({ data: copy });
        });
      }
    }
    static closeAll() { open.clear(); }
  };
}
