# Tool Description Revision Notes

## Revision 1 — search_tickets

### Before

Search tickets.

### After

Search previous support tickets for relevant examples.
Use category when the ticket type is known.

### Observed behavioral change

Pending agent run. This evidence will be completed after
the hand-written agent loop is implemented and the model
is run with the revised description.

### Why this change was made

The revised description tells the model what the tool is
useful for and when the optional category filter is useful,
without putting implementation details or business logic
into the description.