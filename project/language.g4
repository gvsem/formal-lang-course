grammar language;

WS: [ \r\n\t]+ -> skip;

VAR: [a-z] [a-z_"0-9]*;
NUM: '0' | ([1-9][0-9]*);
CHAR: '\u0022' [a-z] '\u0022';

prog: stmt* EOF;

stmt: bind | add | remove | declare;

declare: 'let' VAR 'is' 'graph';

bind: 'let' VAR '=' expr;

remove:
	'remove' ('vertex' | 'edge' | 'vertices') expr 'from' VAR;

add: 'add' ('vertex' | 'edge') expr 'to' VAR;

expr:
	NUM			# Expr_num
	| CHAR		# Expr_char
	| VAR		# Expr_var
	| edge_expr	# Expr_edge_expr
	| set_expr	# Expr_set_expr
	| regexp	# Expr_regexp
	| select	# Expr_select;

set_expr: '[' expr (',' expr)* ']';

edge_expr: '(' expr ',' expr ',' expr ')';

regexp:
	CHAR				# Regex_char
	| VAR				# Regex_var
	| '(' regexp ')'	# Regex_braces
	| regexp '|' regexp	# Regex_union
	| regexp '^' range	# Regex_repeat
	| regexp '.' regexp	# Regex_concat
	| regexp '&' regexp	# Regex_intersect;

range: '[' NUM '..' NUM? ']';

select:
	v_filter? v_filter? 'return' VAR (',' VAR)? 'where' VAR 'reachable' 'from' VAR 'in' VAR 'by'
		expr;

v_filter: 'for' VAR 'in' expr;
