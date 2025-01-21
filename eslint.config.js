import eslintPluginNode from 'eslint-plugin-node'; // Import the plugin as an object

export default [
  {
    languageOptions: {
      ecmaVersion: 2020,
      sourceType: 'module',  // For ES modules
    },
    rules: {
      'no-console': 'warn',
      'no-unused-vars': 'error',
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
