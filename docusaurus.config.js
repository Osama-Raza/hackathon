// @ts-check
// `@type` JSDoc annotations allow IDEs and type checkers to type-check this file
// to check it on save. You don't need these across your project, but it's nice for this one.

/** @type {import('@docusaurus/types').DocusaurusConfig} */
(
  module.exports = {
    title: 'Documentation Reproducibility',
    tagline: 'Reproducible Documentation for Physical AI and Robotics',
    url: 'https://your-username.github.io',
    baseUrl: '/',
    onBrokenLinks: 'warn',
    onBrokenMarkdownLinks: 'warn',
    favicon: 'img/favicon.ico',
    organizationName: 'your-username', // Usually your GitHub org/user name.
    projectName: 'hackathon', // Usually your repo name.
    presets: [
      [
        '@docusaurus/preset-classic',
        /** @type {import('@docusaurus/preset-classic').Options} */
        ({
          docs: {
            sidebarPath: require.resolve('./sidebars.js'),
            // Please change this to your repo.
            editUrl: 'https://github.com/facebook/docusaurus/edit/main/website/',
          },
          blog: {
            showReadingTime: true,
            // Please change this to your repo.
            editUrl: 'https://github.com/facebook/docusaurus/edit/main/website/blog/',
          },
          theme: {
            customCss: require.resolve('./src/css/custom.css'),
          },
        }),
      ],
    ],

    themeConfig:
      /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
      ({
        navbar: {
          title: 'Documentation Reproducibility',
          logo: {
            alt: 'Physical AI Logo',
            src: 'img/logo.svg',
          },
          items: [
            {
              type: 'doc',
              docId: 'intro',
              position: 'left',
              label: 'Tutorial',
            },
            {
              href: 'https://github.com/Osama-Raza',
              label: 'GitHub',
              position: 'right',
            },
          ],
        },
        footer: {
          style: 'dark',
          links: [
            {
              title: 'Docs',
              items: [
                {
                  label: 'Tutorial',
                  to: '/docs/intro',
                },
              ],
            },
            {
              title: 'More',
              items: [
                {
                  label: 'GitHub',
                  href: 'https://github.com/Osama-Raza',
                },
              ],
            },
          ],
          copyright: `Copyright © ${new Date().getFullYear()} Physical AI Learning Platform. Built with Docusaurus.`,
        },
        prism: {
          theme: require('prism-react-renderer').themes.github,
          darkTheme: require('prism-react-renderer').themes.dracula,
        },
      }),
  }
)