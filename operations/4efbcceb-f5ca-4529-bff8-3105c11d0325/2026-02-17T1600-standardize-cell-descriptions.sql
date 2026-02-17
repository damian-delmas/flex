-- Plan 8.5: Standardize cell descriptions
-- All cells get terse one-line descriptions. @orient handles the rest.

-- axpstack-context (2ac5c1bc)
UPDATE _meta SET value = 'AXP Systems ecosystem documentation.' WHERE key = 'description';

-- claude_chat (d9ca9afd)
UPDATE _meta SET value = 'Claude.ai conversations.' WHERE key = 'description';

-- claude_code (e7547b39)
UPDATE _meta SET value = 'Claude Code sessions.' WHERE key = 'description';

-- flexsearch-context (4efbcceb)
UPDATE _meta SET value = 'FlexSearch codebase documentation.' WHERE key = 'description';

-- inventory (443f6b1f)
UPDATE _meta SET value = 'NPTA inventory ecosystem.' WHERE key = 'description';

-- qmem (17b9ea9f)
UPDATE _meta SET value = 'QMEM codebase documentation.' WHERE key = 'description';

-- thread-codebase (c2c785f9)
UPDATE _meta SET value = 'Thread codebase documentation (superseded by FlexSearch).' WHERE key = 'description';
