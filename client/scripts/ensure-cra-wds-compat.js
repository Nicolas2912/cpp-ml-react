const fs = require("fs");
const path = require("path");

const startScriptPath = path.join(
  __dirname,
  "..",
  "node_modules",
  "react-scripts",
  "scripts",
  "start.js"
);

const devServerConfigPath = path.join(
  __dirname,
  "..",
  "node_modules",
  "react-scripts",
  "config",
  "webpackDevServer.config.js"
);

function patchStartScript() {
  if (!fs.existsSync(startScriptPath)) {
    console.warn(
      "[ensure-cra-wds-compat] react-scripts start script not found; skipping"
    );
    return;
  }

  const legacyMarker =
    "    const devServer = new WebpackDevServer(serverConfig, compiler);";
  const blockStart = "    const webpackDevServerVersion = (() => {";
  const blockEnd =
    "        : new WebpackDevServer(compiler, serverConfig);";
  const canonicalBlock = [
    "    const webpackDevServerVersion = (() => {",
    "      try {",
    "        return require(\"webpack-dev-server/package.json\").version;",
    "      } catch (error) {",
    "        return \"0.0.0\";",
    "      }",
    "    })();",
    "    const devServer =",
    "      parseInt(webpackDevServerVersion.split(\".\")[0], 10) >= 5",
    "        ? new WebpackDevServer(serverConfig, compiler)",
    "        : new WebpackDevServer(compiler, serverConfig);",
  ].join("\n");

  let fileContents = fs.readFileSync(startScriptPath, "utf8");

  const escapeForRegex = (value) =>
    value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

  const blockRegex = new RegExp(
    `${escapeForRegex(blockStart)}[\\s\\S]*?${escapeForRegex(blockEnd)}\n?`,
    "g"
  );

  if (blockRegex.test(fileContents)) {
    fileContents = fileContents.replace(blockRegex, `${canonicalBlock}\n`);
  } else if (fileContents.includes(legacyMarker)) {
    fileContents = fileContents.replace(legacyMarker, canonicalBlock);
  } else if (!fileContents.includes(canonicalBlock)) {
    console.warn(
      "[ensure-cra-wds-compat] Unable to locate webpack-dev-server constructor block"
    );
    return;
  }

  fileContents = fileContents.replace(
    /(\n        : new WebpackDevServer\(compiler, serverConfig\);)+/g,
    "\n        : new WebpackDevServer(compiler, serverConfig);"
  );

  fs.writeFileSync(startScriptPath, fileContents);
}

function patchDevServerConfig() {
  if (!fs.existsSync(devServerConfigPath)) {
    console.warn(
      "[ensure-cra-wds-compat] react-scripts webpackDevServer.config.js not found; skipping"
    );
    return;
  }

  const legacyBlock = [
    "    onBeforeSetupMiddleware(devServer) {",
    "      // Keep `evalSourceMapMiddleware`",
    "      // middlewares before `redirectServedPath` otherwise will not have any effect",
    "      // This lets us fetch source contents from webpack for the error overlay",
    "      devServer.app.use(evalSourceMapMiddleware(devServer));",
    "",
    "      if (fs.existsSync(paths.proxySetup)) {",
    "        // This registers user provided middleware for proxy reasons",
    "        require(paths.proxySetup)(devServer.app);",
    "      }",
    "    },",
    "    onAfterSetupMiddleware(devServer) {",
    "      // Redirect to `PUBLIC_URL` or `homepage` from `package.json` if url not match",
    "      devServer.app.use(redirectServedPath(paths.publicUrlOrPath));",
    "",
    "      // This service worker file is effectively a 'no-op' that will reset any",
    "      // previous service worker registered for the same host:port combination.",
    "      // We do this in development to avoid hitting the production cache if",
    "      // it used the same host and port.",
    "      // https://github.com/facebook/create-react-app/issues/2272#issuecomment-302832432",
    "      devServer.app.use(noopServiceWorkerMiddleware(paths.publicUrlOrPath));",
    "    },",
  ].join("\n");

  const modernBlock = [
    "    setupMiddlewares(middlewares, devServer) {",
    "      if (!devServer) {",
    "        throw new Error(\"webpack-dev-server is not defined\");",
    "      }",
    "",
    "      middlewares.unshift(evalSourceMapMiddleware(devServer));",
    "",
    "      if (fs.existsSync(paths.proxySetup)) {",
    "        require(paths.proxySetup)(devServer.app);",
    "      }",
    "",
    "      middlewares.push(redirectServedPath(paths.publicUrlOrPath));",
    "      middlewares.push(noopServiceWorkerMiddleware(paths.publicUrlOrPath));",
    "",
    "      return middlewares;",
    "    },",
  ].join("\n");

  let fileContents = fs.readFileSync(devServerConfigPath, "utf8");

  if (fileContents.includes("setupMiddlewares")) {
    // Already patched block
  } else if (fileContents.includes(legacyBlock)) {
    fileContents = fileContents.replace(legacyBlock, modernBlock);
  } else {
    console.warn(
      "[ensure-cra-wds-compat] Unable to locate legacy devServer middleware hooks"
    );
  }

  const httpsLine = "    https: getHttpsConfig(),";
  if (fileContents.includes(httpsLine) && !fileContents.includes("server: (() => {")) {
    const httpsReplacement = [
      "    server: (() => {",
      "      const httpsConfig = getHttpsConfig();",
      "",
      "      if (httpsConfig) {",
      "        return { type: \"https\", options: httpsConfig };",
      "      }",
      "",
      "      return { type: \"http\" };",
      "    })(),",
    ].join("\n");

    fileContents = fileContents.replace(httpsLine, httpsReplacement);
  }

  fs.writeFileSync(devServerConfigPath, fileContents);
}

patchStartScript();
patchDevServerConfig();
console.info(
  "[ensure-cra-wds-compat] Patched react-scripts for webpack-dev-server compatibility"
);
