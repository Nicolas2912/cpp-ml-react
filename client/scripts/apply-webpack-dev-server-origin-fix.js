const fs = require("fs");
const path = require("path");
const url = require("url");

const serverPath = path.join(
  __dirname,
  "..",
  "node_modules",
  "webpack-dev-server",
  "lib",
  "Server.js"
);

if (!fs.existsSync(serverPath)) {
  console.warn("[apply-webpack-dev-server-origin-fix] Server.js not found; skipping");
  process.exit(0);
}

const fileContents = fs.readFileSync(serverPath, "utf8");

if (fileContents.includes("hostHeaderValue")) {
  // Patch already applied
  process.exit(0);
}

const searchSnippet =
  "    const isValidHostname =\n" +
  "      (hostname !== null && ipaddr.IPv4.isValid(hostname)) ||\n" +
  "      (hostname !== null && ipaddr.IPv6.isValid(hostname)) ||\n" +
  '      hostname === "localhost" ||\n' +
  "      (hostname !== null && hostname.endsWith(\".localhost\")) ||\n" +
  "      hostname === this.options.host;\n\n" +
  "    if (isValidHostname) {\n" +
  "      return true;\n" +
  "    }\n";

const replacementSnippet =
  "    const isHostnameIP =\n" +
  "      (hostname !== null && ipaddr.IPv4.isValid(hostname)) ||\n" +
  "      (hostname !== null && ipaddr.IPv6.isValid(hostname));\n\n" +
  "    if (isHostnameIP) {\n" +
  '      if (headerToCheck === "origin") {\n' +
  "        const hostHeaderValue = headers.host;\n\n" +
  "        if (!hostHeaderValue) {\n" +
  "          return false;\n" +
  "        }\n\n" +
  "        const hostHostname = url.parse(\n" +
  "          /^(.+:)?\\/\\//.test(hostHeaderValue)\n" +
  "            ? hostHeaderValue\n" +
  "            : `//${hostHeaderValue}`,\n" +
  "          false,\n" +
  "          true\n" +
  "        ).hostname;\n\n" +
  "        if (hostHostname !== hostname) {\n" +
  "          return false;\n" +
  "        }\n" +
  "      }\n\n" +
  "      return true;\n" +
  "    }\n\n" +
  '    const isValidHostname =\n' +
  '      hostname === "localhost" ||\n' +
  "      (hostname !== null && hostname.endsWith(\".localhost\")) ||\n" +
  "      hostname === this.options.host;\n\n" +
  "    if (isValidHostname) {\n" +
  "      return true;\n" +
  "    }\n";

if (!fileContents.includes(searchSnippet)) {
  console.warn("[apply-webpack-dev-server-origin-fix] Expected snippet not found; skipping");
  process.exit(0);
}

const updatedContents = fileContents.replace(searchSnippet, replacementSnippet);

fs.writeFileSync(serverPath, updatedContents);

console.info("[apply-webpack-dev-server-origin-fix] Patched webpack-dev-server origin validation");
