// @ts-check
// `@type` JSDoc annotations allow IDEs and type checkers to type-check this file
// to check it on save. You don't need these across your project, but it's nice for this one.

/** @type {import('@docusaurus/types').DocusaurusConfig} */
(
  module.exports = {
    title: 'Documentation Reproducibility',
    tagline: 'Reproducible Documentation for Physical AI and Robotics',
    url: process.env.DEPLOYMENT_ENV === 'vercel' ? 'https://hackathon-virid-zeta.vercel.app' : 'https://osama-raza.github.io',
    baseUrl: process.env.DEPLOYMENT_ENV === 'vercel' ? '/' : '/hackathon/',
    onBrokenLinks: 'warn',
    onBrokenMarkdownLinks: 'warn', // TODO: migrate to markdown.hooks.onBrokenMarkdownLinks in Docusaurus v4
    favicon: 'img/favicon.ico',
    organizationName: 'Osama-Raza', // Usually your GitHub org/user name.
    projectName: 'hackathon', // Usually your repo name.
    i18n: {
      defaultLocale: 'en',
      locales: ['en', 'ur'],
      localeConfigs: {
        en: {
          label: 'English',
        },
        ur: {
          label: 'اردو',
          direction: 'rtl',
        },
      },
    },
    presets: [
      [
        '@docusaurus/preset-classic',
        /** @type {import('@docusaurus/preset-classic').Options} */
        ({
          docs: {
            sidebarPath: require.resolve('./sidebars.js'),
            editUrl: 'https://github.com/Osama-Raza/hackathon/edit/main/',
          },
          blog: {
            showReadingTime: false,
            editUrl: 'https://github.com/Osama-Raza/hackathon/edit/main/',
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
            src: 'img/logo.jpeg',
          },
          items: [
            {
              type: 'doc',
              docId: 'intro',
              position: 'left',
              label: 'Tutorial',
            },
            {
              type: 'localeDropdown',
              position: 'right',
            },
            {
              href: 'https://github.com/Osama-Raza/hackathon',
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