import eslintPluginNode from 'eslint-plugin-node'; // Import the plugin as an object

export default [
  {
    languageOptions: {
      ecmaVersion: 2020,
      sourceType: 'module',  // For ES modules
    },
    rules: {
      'no-unused-vars': ['error', { args: 'none', ignoreRestSiblings: true }], // Error on unused vars, but allow unused function arguments
      'no-debugger': 'error', // Disallow `debugger` statements
      'eqeqeq': ['error', 'always'], // Enforce strict equality
      'curly': 'error', // Enforce curly braces around blocks
      'no-var': 'error', // Disallow `var` in favor of `let` and `const`
      'prefer-const': 'error', // Suggest using `const` wherever possible
      'arrow-spacing': ['error', { before: true, after: true }], // Enforce spacing around arrow functions
      'semi': ['error', 'always'], // Enforce semicolons
      'quotes': ['error', 'single', { avoidEscape: true }], // Enforce single quotes, allow escaping
      'indent': ['error', 2], // Enforce 2-space indentation
      'comma-dangle': ['error', 'always-multiline'], // Require trailing commas for multiline constructs
      'object-curly-spacing': ['error', 'always'], // Require spacing inside curly braces
    },
  },

  // Add the plugin to the config in the correct format
  {
    plugins: {
      node: eslintPluginNode,  // Pass the plugin object here
    },
    rules: {
      'node/no-missing-import': 'error',  // Example rule from the plugin
    },
  },
];
