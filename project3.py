#######################################
# IMPORTS
#######################################

from strings_with_arrows import *

import string   # for the used of the string char
import os
import math

#######################################
# CONSTANTS
#######################################

#  purpose : constant letter and number need for creation of the compiler

DIGITS = '0123456789'
LEToken_TYPEERS = string.ascii_letters
LEToken_TYPEERS_DIGITS = LEToken_TYPEERS + DIGITS

#######################################
# ERRORS
#######################################

#  purpose : this class is used to keep track of the error while running the interpteter
class Error:
  def __init__(self, startIdx, endIdx, error_name, details):
    self.startIdx = startIdx
    self.endIdx = endIdx
    self.error_name = error_name
    self.details = details

  def as_string(self):
    RESULT  = f'{self.error_name}: {self.details}\n'
    RESULT += f'File {self.startIdx.fn}, line {self.startIdx.ln + 1}'
    RESULT += '\n\n' + string_with_arrows(self.startIdx.ftxt, self.startIdx, self.endIdx)
    return RESULT

#  purpose : error while in the Lexer process
class IllegalCharError(Error):
  def __init__(self, startIdx, endIdx, details):
    super().__init__(startIdx, endIdx, 'Illegal Character', details)

class ExpectedCharError(Error):
  def __init__(self, startIdx, endIdx, details):
    super().__init__(startIdx, endIdx, 'Expected Character', details)

#  purpose : error while in the parser process
class InvalidSyntaxError(Error):
  def __init__(self, startIdx, endIdx, details=''):
    super().__init__(startIdx, endIdx, 'Invalid Syntax', details)

# purpose : to keep track of run time error
class RTError(Error):
  def __init__(self, startIdx, endIdx, details, context):
    super().__init__(startIdx, endIdx, 'Runtime Error', details)
    self.context = context

  def as_string(self):
    RESULT  = self.generate_traceback()
    RESULT += f'{self.error_name}: {self.details}'
    RESULT += '\n\n' + string_with_arrows(self.startIdx.ftxt, self.startIdx, self.endIdx)
    return RESULT

  def generate_traceback(self):
    RESULT = ''
    pos = self.startIdx
    ctx = self.context

    while ctx:
      RESULT = f'  File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.display_name}\n' + RESULT
      pos = ctx.parent_entry_pos
      ctx = ctx.parent

    return 'Traceback (most recent call last):\n' + RESULT

#######################################
# POSITION
#######################################

# purpose : to keep track of the position while using the interpreter

class Position:
  def __init__(self, idx, ln, col, fn, ftxt):
    self.idx = idx
    self.ln = ln
    self.col = col
    self.fn = fn
    self.ftxt = ftxt

  def increment(self, current_char=None):
# if it is in the same line, keep increment the idx and col by 1
    self.idx += 1
    self.col += 1

# if the newLine is encounter
# the colum start at 0, and new line is incremnt
    if current_char == '\n':
      self.ln += 1
      self.col = 0

    return self

  def clone(self):
    return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

#######################################
# TOKENS
#######################################

# purpose : to assgn the token size and its Value

Token_type_INT			= 'INT'
Token_type_FLOAT    	= 'FLOAT'
Token_type_STRING		= 'STRING'
Token_type_IDENTIFIER	= 'IDENTIFIER'
Token_type_KEYWORD		= 'KEYWORD'
Token_type_PLUS     	= 'PLUS'
Token_type_MINUS    	= 'MINUS'
Token_type_MUL      	= 'MUL'
Token_type_DIV      	= 'DIV'
Token_type_POW			= 'POW'
Token_type_EQ			= 'EQ'
Token_type_LPAREN   	= 'LPAREN'
Token_type_RPAREN   	= 'RPAREN'
Token_type_LSQUARE      = 'LSQUARE'
Token_type_RSQUARE      = 'RSQUARE'
Token_type_EE			= 'EE'
Token_type_NE			= 'NE'
Token_type_LT			= 'LT'
Token_type_GT			= 'GT'
Token_type_LTE			= 'LTE'
Token_type_GTE			= 'GTE'
Token_type_COMMA		= 'COMMA'
Token_type_ARROW		= 'ARROW'
Token_type_NEWLINE		= 'NEWLINE'
Token_type_EOF			= 'EOF'

KEYWORDS = [
  'get',
  'and',
  'or',
  'not',
  'if',
  'elif',
  'else',
  'for',
  'to',
  'while',
  'fun',
  'then',
  'end'
]

class Token:
  def __init__(self, type_, value=None, startIdx=None, endIdx=None):
    self.type = type_   # assign its type
    self.value = value  # assign its value

    if startIdx:
      self.startIdx = startIdx.clone()
      self.endIdx = startIdx.clone()
      self.endIdx.increment()       # position end = position start + 1

    if endIdx:
      self.endIdx = endIdx.clone()

# purpose : to return its type and value, in case there are type and value
  def matches(self, type_, value):
    return self.type == type_ and self.value == value

  def __repr__(self):
    if self.value: return f'{self.type}:{self.value}'
    return f'{self.type}'

# ---------------------------------------------------------------------------------------------------------#
#######################################
                # LEXER
#######################################

# purpose : to break the text into token

class Lexer:
  def __init__(self, fn, text):
    self.fn = fn
    self.text = text
    self.pos = Position(-1, 0, -1, fn, text)
    self.current_char = None
    self.increment()

# purpose : to increment the position in the text
  def increment(self):
    self.pos.increment(self.current_char)
    self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

# purpose : to create the token from the input,
#           and append the token and its type
  def create_tokenens(self):
    tokens = []     # to keep the token

    while self.current_char != None:    # traverse from all of the token in terminal
      if self.current_char in ' \t':    # if the current char is not TAB
        self.increment()
      elif self.current_char in ';\n':  # if the current char is not newline
        tokens.append(Token(Token_type_NEWLINE, startIdx=self.pos))
        self.increment()
      elif self.current_char in DIGITS:
        tokens.append(self.create_number())
      elif self.current_char in LEToken_TYPEERS:
        tokens.append(self.create_identifier())
      elif self.current_char == '"':    # string start with double quote
        tokens.append(self.create_string())
      elif self.current_char == '+':
        tokens.append(Token(Token_type_PLUS, startIdx=self.pos))
        self.increment()
      elif self.current_char == '-':
        tokens.append(self.create_minus_or_arrow())
      elif self.current_char == '*':
        tokens.append(Token(Token_type_MUL, startIdx=self.pos))
        self.increment()
      elif self.current_char == '/':
        tokens.append(Token(Token_type_DIV, startIdx=self.pos))
        self.increment()
      elif self.current_char == '^':
        tokens.append(Token(Token_type_POW, startIdx=self.pos))
        self.increment()
      elif self.current_char == '(':
        tokens.append(Token(Token_type_LPAREN, startIdx=self.pos))
        self.increment()
      elif self.current_char == ')':
        tokens.append(Token(Token_type_RPAREN, startIdx=self.pos))
        self.increment()
      elif self.current_char == '[':
        tokens.append(Token(Token_type_LSQUARE, startIdx=self.pos))
        self.increment()
      elif self.current_char == ']':
        tokens.append(Token(Token_type_RSQUARE, startIdx=self.pos))
        self.increment()
      elif self.current_char == '!':
        token, error = self.create_notEqual()
        if error: return [], error
        tokens.append(token)
      elif self.current_char == '=':
        tokens.append(self.create_equal())
      elif self.current_char == '<':
        tokens.append(self.create_lessThan())
      elif self.current_char == '>':
        tokens.append(self.create_greaterThan())
      elif self.current_char == ',':
        tokens.append(Token(Token_type_COMMA, startIdx=self.pos))
        self.increment()
      else:
        startIdx = self.pos.clone()     # in case there is error
        char = self.current_char
        self.increment()                # return the error character
        return [], IllegalCharError(startIdx, self.pos, "'" + char + "'")

    tokens.append(Token(Token_type_EOF, startIdx=self.pos))
    return tokens, None

# purpose :  to create number
# since the number can be both interger or float
  def create_number(self):
    num_str = ''
    dot_count = 0   # in case of float, the number has the dot
    startIdx = self.pos.clone()

    while self.current_char != None and self.current_char in DIGITS + '.':
      if self.current_char == '.':
        if dot_count == 1: break    # the number can have only 1 dot
        dot_count += 1
      num_str += self.current_char
      self.increment()

    if dot_count == 0:  # if there is no dot, it is int
      return Token(Token_type_INT, int(num_str), startIdx, self.pos)
    else:               # if there is dot, it is float
      return Token(Token_type_FLOAT, float(num_str), startIdx, self.pos)

# purpose : to create string type token
  def create_string(self):
    string = ''     # to keep the string
    startIdx = self.pos.clone()
    escape_character = False
    self.increment()

    escape_characters = {
      'n': '\n',
      't': '\t'
    }

    # if there is the character and it is not the double quote and the escape character
    while self.current_char != None and (self.current_char != '"' or escape_character):
      if escape_character:
        string += escape_characters.get(self.current_char, self.current_char)
      else:
        if self.current_char == '\\':   # back slash for the purpose of the escape char
          escape_character = True
        else:
          string += self.current_char
      self.increment()
      escape_character = False       # at the end of every loop, reset the double quote to false

    self.increment()
    return Token(Token_type_STRING, string, startIdx, self.pos)

# purpose : to create identifier type token
  def create_identifier(self):
    id_str = ''
    startIdx = self.pos.clone()

    while self.current_char != None and self.current_char in LEToken_TYPEERS_DIGITS + '_':
      id_str += self.current_char
      self.increment()

    # token type is the keyword if it is in the keyword category
    tok_type = Token_type_KEYWORD if id_str in KEYWORDS else Token_type_IDENTIFIER
    return Token(tok_type, id_str, startIdx, self.pos)

  def create_minus_or_arrow(self):
    tok_type = Token_type_MINUS
    startIdx = self.pos.clone()
    self.increment()

    if self.current_char == '>':
      self.increment()
      tok_type = Token_type_ARROW

    return Token(tok_type, startIdx=startIdx, endIdx=self.pos)

# purpose : to create not equal type token
  def create_notEqual(self):
    startIdx = self.pos.clone()
    self.increment()

    if self.current_char == '=':
      self.increment()
      return Token(Token_type_NE, startIdx=startIdx, endIdx=self.pos), None

    self.increment()
    return None, ExpectedCharError(startIdx, self.pos, "'=' (after '!')")

# purpose : to create equal type token
  def create_equal(self):
    tok_type = Token_type_EQ
    startIdx = self.pos.clone()
    self.increment()

    if self.current_char == '=':
      self.increment()
      tok_type = Token_type_EE      # assign the toke type

    return Token(tok_type, startIdx=startIdx, endIdx=self.pos)

# purpose : to create less than type token
  def create_lessThan(self):
    tok_type = Token_type_LT
    startIdx = self.pos.clone()
    self.increment()

    if self.current_char == '=':
      self.increment()
      tok_type = Token_type_LTE     # assign the toke type

    return Token(tok_type, startIdx=startIdx, endIdx=self.pos)

# purpose : to create greater than type token
  def create_greaterThan(self):
    tok_type = Token_type_GT
    startIdx = self.pos.clone()
    self.increment()

    if self.current_char == '=':
      self.increment()
      tok_type = Token_type_GTE     # assign the toke type

    return Token(tok_type, startIdx=startIdx, endIdx=self.pos)


# ---------------------------------------------------------------------------------------------------------#

#######################################
# NODES
#######################################

# purpose : the node is create for the purpose of the Parser
#           since the Paerser will consider from the arrangment of the node
#           from building of the tree

class NumberNode:
  def __init__(self, tok):
    self.tok = tok

    self.startIdx = self.tok.startIdx
    self.endIdx = self.tok.endIdx

  def __repr__(self):
    return f'{self.tok}'    # return the token in the tree

class StringNode:
  def __init__(self, tok):
    self.tok = tok

    self.startIdx = self.tok.startIdx
    self.endIdx = self.tok.endIdx

  def __repr__(self):
    return f'{self.tok}'

class ListNode:
  def __init__(self, element_nodes, startIdx, endIdx):
    self.element_nodes = element_nodes

    self.startIdx = startIdx
    self.endIdx = endIdx

class VarAccessNode:
  def __init__(self, var_name_token):
    self.var_name_token = var_name_token

    self.startIdx = self.var_name_token.startIdx
    self.endIdx = self.var_name_token.endIdx

class VarAssignNode:
  def __init__(self, var_name_token, value_node):
    self.var_name_token = var_name_token
    self.value_node = value_node

    self.startIdx = self.var_name_token.startIdx
    self.endIdx = self.value_node.endIdx

# node.left  opperator node.right
class BinaryOperationNode:
  def __init__(self, left_node, operator_tokenen, right_node):
    self.left_node = left_node
    self.operator_tokenen = operator_tokenen
    self.right_node = right_node
    self.startIdx = self.left_node.startIdx
    self.endIdx = self.right_node.endIdx

  def __repr__(self):
    return f'({self.left_node}, {self.operator_tokenen}, {self.right_node})'

# operation node
class UnaryOperationNode:
  def __init__(self, operator_tokenen, node):
    self.operator_tokenen = operator_tokenen
    self.node = node

    self.startIdx = self.operator_tokenen.startIdx
    self.endIdx = node.endIdx

  def __repr__(self):
    return f'({self.operator_tokenen}, {self.node})'

class IfNode:
  def __init__(self, cases, else_case):
    self.cases = cases
    self.else_case = else_case

    self.startIdx = self.cases[0][0].startIdx
    self.endIdx = (self.else_case or self.cases[len(self.cases) - 1])[0].endIdx

class WhileNode:
  def __init__(self, condition_node, body_node, should_return_null):
    self.condition_node = condition_node
    self.body_node = body_node
    self.should_return_null = should_return_null

    self.startIdx = self.condition_node.startIdx
    self.endIdx = self.body_node.endIdx

# purpose :
class FuncDefNode:
  def __init__(self, var_name_token, arg_name_tokens, body_node, should_return_null):
    self.var_name_token = var_name_token
    self.arg_name_tokens = arg_name_tokens
    self.body_node = body_node
    self.should_return_null = should_return_null

    if self.var_name_token:     # if the function has the name
      self.startIdx = self.var_name_token.startIdx  # set the start idx to
    elif len(self.arg_name_tokens) > 0:
      self.startIdx = self.arg_name_tokens[0].startIdx
    else:
      self.startIdx = self.body_node.startIdx

    self.endIdx = self.body_node.endIdx

# purpose : take the node we call and the arge node that pass in to the function
class CallNode:
  def __init__(self, node_to_call, arg_nodes):
    self.node_to_call = node_to_call
    self.arg_nodes = arg_nodes

    self.startIdx = self.node_to_call.startIdx

    if len(self.arg_nodes) > 0:
      self.endIdx = self.arg_nodes[len(self.arg_nodes) - 1].endIdx
    else:
      self.endIdx = self.node_to_call.endIdx

# ---------------------------------------------------------------------------------------------------------#

#######################################
# PARSE RESULT
#######################################

# purpose : to keep track of the node RESULT
#           insted of reuturning the node

# this class is to keep track of the error and the node
class ParseResult:
  def __init__(self):
    self.error = None
    self.node = None
    self.last_registered_increment_count = 0
    self.increment_count = 0
    self.to_reverse_count = 0

  def register_incrementment(self):
    self.last_registered_increment_count = 1
    self.increment_count += 1

 # check for the paerse result whether there is the error or not
  def register(self, result):
    self.last_registered_increment_count = result.increment_count
    self.increment_count += result.increment_count
    if result.error: self.error = result.error  # return if there is the error
    return result.node  # return the result

  def try_register(self, result):
    if result.error:
      self.to_reverse_count = result.increment_count
      return None
    return self.register(result)

  def success(self, node):
    self.node = node
    return self

    # take the error and return the error
  def failure(self, error):
    if not self.error or self.last_registered_increment_count == 0:
      self.error = error
    return self

# ---------------------------------------------------------------------------------------------------------#

#######################################
# PARSER
#######################################

# purpose :  This is syntax tree of the program
#            the perser will keep track of  the token index, similary to the lexer
#            this is used to check whether the input token follow the grammar rule
#            generated by the tree or not
#            if we put sth in correct, return the in correct

class Parser:
  def __init__(self, tokens):
    self.tokens = tokens
    self.tok_idx = -1
    self.increment()

  def increment(self):
    self.tok_idx += 1
    self.update_current_token()
    return self.current_token

  def reverse(self, amount=1):
    self.tok_idx -= amount
    self.update_current_token()
    return self.current_token

  def update_current_token(self):
    if self.tok_idx >= 0 and self.tok_idx < len(self.tokens):
      self.current_token = self.tokens[self.tok_idx]

# to call for the expression
  def parse(self):
    result = self.statements()
    if not result.error and self.current_token.type != Token_type_EOF:
      return result.failure(InvalidSyntaxError(
        self.current_token.startIdx, self.current_token.endIdx,
        "Expected '+', '-', '*', '/', '^', '==', '!=', '<', '>', <=', '>=', 'and' or 'or'"
      ))
    return result

  ###################################

  def statements(self):
    result = ParseResult()
    statements = []
    startIdx = self.current_token.startIdx.clone()

    while self.current_token.type == Token_type_NEWLINE:
      result.register_incrementment()
      self.increment()

    statement = result.register(self.expr())
    if result.error: return result
    statements.append(statement)

    more_statements = True

    while True:
      newline_count = 0
      while self.current_token.type == Token_type_NEWLINE:
        result.register_incrementment()
        self.increment()
        newline_count += 1
      if newline_count == 0:
        more_statements = False

      if not more_statements: break
      statement = result.try_register(self.expr())
      if not statement:
        self.reverse(result.to_reverse_count)
        more_statements = False
        continue
      statements.append(statement)

    return result.success(ListNode(
      statements,
      startIdx,
      self.current_token.endIdx.clone()
    ))

# KEYWORD:VAR IDENTIFITER EQ expr
  def expr(self):
    result = ParseResult()

# the order of expression is matter
# the order of expression as
    # KEYWORD:VAR IDENTIFITER EQ expr

    # 1 : check for KEYOWORD OR int
    if self.current_token.matches(Token_type_KEYWORD, 'get'):
      result.register_incrementment()
      self.increment()

      # 3 : check for IDENTIFIER, if no => error
      if self.current_token.type != Token_type_IDENTIFIER:
        return result.failure(InvalidSyntaxError(
          self.current_token.startIdx, self.current_token.endIdx,
          "Expected identifier"
        ))

        # 4 : var name
      var_name = self.current_token
      result.register_incrementment()
      self.increment()

      if self.current_token.type != Token_type_EQ:
        return result.failure(InvalidSyntaxError(
          self.current_token.startIdx, self.current_token.endIdx,
          "Expected '='"
        ))

      result.register_incrementment()
      self.increment()
      expr = result.register(self.expr())
      if result.error: return result
      return result.success(VarAssignNode(var_name, expr))

    node = result.register(self.binary_operation(self.comp_expr, ((Token_type_KEYWORD, 'and'), (Token_type_KEYWORD, 'or'))))

    if result.error:
      return result.failure(InvalidSyntaxError(
        self.current_token.startIdx, self.current_token.endIdx,
        "Expected 'get', 'if', 'for', 'while', 'fun', int, float, identifier, '+', '-', '(', '[' or 'not'"
      ))

    return result.success(node)

  def comp_expr(self):
    result = ParseResult()

    if self.current_token.matches(Token_type_KEYWORD, 'not'):
      operator_tokenen = self.current_token
      result.register_incrementment()
      self.increment()

      node = result.register(self.comp_expr())
      if result.error: return result
      return result.success(UnaryOperationNode(operator_tokenen, node))

    node = result.register(self.binary_operation(self.arith_expr, (Token_type_EE, Token_type_NE, Token_type_LT, Token_type_GT, Token_type_LTE, Token_type_GTE)))

    if result.error:
      return result.failure(InvalidSyntaxError(
        self.current_token.startIdx, self.current_token.endIdx,
        "Expected int, float, identifier, '+', '-', '(', '[', 'if', 'for', 'while', 'fun' or 'not'"
      ))

    return result.success(node)

  def arith_expr(self):
    return self.binary_operation(self.term, (Token_type_PLUS, Token_type_MINUS))

  def term(self):
    return self.binary_operation(self.factor, (Token_type_MUL, Token_type_DIV))

# look for the integer or FLOAT
# and return the number node of that tokens
  def factor(self):
    result = ParseResult()
    tok = self.current_token    # get the current token

    if tok.type in (Token_type_PLUS, Token_type_MINUS):     # check whether the token is plus or minus
      result.register_incrementment()                       # increment to return the number node of that token
      self.increment()
      factor = result.register(self.factor())
      if result.error: return result
      return result.success(UnaryOperationNode(tok, factor))

    return self.power()

  def power(self):
    return self.binary_operation(self.call, (Token_type_POW, ), self.factor)


# purpose : this is to call the function
  def call(self):
    result = ParseResult()
    atom = result.register(self.atom())
    if result.error: return result

    if self.current_token.type == Token_type_LPAREN:
      result.register_incrementment()
      self.increment()
      arg_nodes = []

      if self.current_token.type == Token_type_RPAREN:
        result.register_incrementment()
        self.increment()
      else:
        arg_nodes.append(result.register(self.expr()))
        if result.error:
          return result.failure(InvalidSyntaxError(
            self.current_token.startIdx, self.current_token.endIdx,
            "Expected ')', 'get', 'if', 'for', 'while', 'fun', int, float, identifier, '+', '-', '(', '[' or 'not'"
          ))

        while self.current_token.type == Token_type_COMMA:
          result.register_incrementment()
          self.increment()

          arg_nodes.append(result.register(self.expr()))
          if result.error: return result

        if self.current_token.type != Token_type_RPAREN:
          return result.failure(InvalidSyntaxError(
            self.current_token.startIdx, self.current_token.endIdx,
            f"Expected ',' or ')'"
          ))

        result.register_incrementment()
        self.increment()
      return result.success(CallNode(atom, arg_nodes))
    return result.success(atom)

  def atom(self):
    result = ParseResult()
    tok = self.current_token

    # 1 :  check for float case
    if tok.type in (Token_type_INT, Token_type_FLOAT):
      result.register_incrementment()
      self.increment()
      return result.success(NumberNode(tok))
    # 2 :  check for string cae
    elif tok.type == Token_type_STRING:
      result.register_incrementment()
      self.increment()
      return result.success(StringNode(tok))
    # 3 : check for identifier case
    elif tok.type == Token_type_IDENTIFIER:
      result.register_incrementment()
      self.increment()
      return result.success(VarAccessNode(tok))

    elif tok.type == Token_type_LPAREN:
      result.register_incrementment()
      self.increment()
      expr = result.register(self.expr())
      if result.error: return result
      if self.current_token.type == Token_type_RPAREN:
        result.register_incrementment()
        self.increment()
        return result.success(expr)
      else:
        return result.failure(InvalidSyntaxError(
          self.current_token.startIdx, self.current_token.endIdx,
          "Expected ')'"
        ))

    elif tok.type == Token_type_LSQUARE:
      list_expr = result.register(self.list_expr())
      if result.error: return result
      return result.success(list_expr)

    elif tok.matches(Token_type_KEYWORD, 'if'):
      if_expr = result.register(self.if_expr())
      if result.error: return result
      return result.success(if_expr)

    elif tok.matches(Token_type_KEYWORD, 'for'):
      for_expr = result.register(self.for_expr())
      if result.error: return result
      return result.success(for_expr)

    elif tok.matches(Token_type_KEYWORD, 'while'):
      while_expr = result.register(self.while_expr())
      if result.error: return result
      return result.success(while_expr)

    elif tok.matches(Token_type_KEYWORD, 'fun'):
      func_def = result.register(self.func_def())
      if result.error: return result
      return result.success(func_def)

    return result.failure(InvalidSyntaxError(
      tok.startIdx, tok.endIdx,
      "Expected int, float, identifier, '+', '-', '(', '[', IF', 'for', 'while', 'fun'"
    ))

  def list_expr(self):
    result = ParseResult()
    element_nodes = []
    startIdx = self.current_token.startIdx.clone()

    if self.current_token.type != Token_type_LSQUARE:
      return result.failure(InvalidSyntaxError(
        self.current_token.startIdx, self.current_token.endIdx,
        f"Expected '['"
      ))

    result.register_incrementment()
    self.increment()

    if self.current_token.type == Token_type_RSQUARE:
      result.register_incrementment()
      self.increment()
    else:
      element_nodes.append(result.register(self.expr()))
      if result.error:
        return result.failure(InvalidSyntaxError(
          self.current_token.startIdx, self.current_token.endIdx,
          "Expected ']', 'get', 'if', 'for', 'while', 'fun', int, float, identifier, '+', '-', '(', '[' or 'not'"
        ))

      while self.current_token.type == Token_type_COMMA:
        result.register_incrementment()
        self.increment()

        element_nodes.append(result.register(self.expr()))
        if result.error: return result

      if self.current_token.type != Token_type_RSQUARE:
        return result.failure(InvalidSyntaxError(
          self.current_token.startIdx, self.current_token.endIdx,
          f"Expected ',' or ']'"
        ))

      result.register_incrementment()
      self.increment()

    return result.success(ListNode(
      element_nodes,
      startIdx,
      self.current_token.endIdx.clone()
    ))

  def if_expr(self):
    result = ParseResult()
    all_cases = result.register(self.if_expr_cases('if'))
    if result.error: return result
    cases, else_case = all_cases
    return result.success(IfNode(cases, else_case))

  def if_expr_b(self):
    return self.if_expr_cases('elif')

  def if_expr_c(self):
    result = ParseResult()
    else_case = None

    if self.current_token.matches(Token_type_KEYWORD, 'else'):
      result.register_incrementment()
      self.increment()

      if self.current_token.type == Token_type_NEWLINE:
        result.register_incrementment()
        self.increment()

        statements = result.register(self.statements())
        if result.error: return result
        else_case = (statements, True)

        if self.current_token.matches(Token_type_KEYWORD, 'end'):
          result.register_incrementment()
          self.increment()
        else:
          return result.failure(InvalidSyntaxError(
            self.current_token.startIdx, self.current_token.endIdx,
            "Expected 'end'"
          ))
      else:
        expr = result.register(self.expr())
        if result.error: return result
        else_case = (expr, False)

    return result.success(else_case)

  def if_expr_b_or_c(self):
    result = ParseResult()
    cases, else_case = [], None

    if self.current_token.matches(Token_type_KEYWORD, 'elif'):
      all_cases = result.register(self.if_expr_b())
      if result.error: return result
      cases, else_case = all_cases
    else:
      else_case = result.register(self.if_expr_c())
      if result.error: return result

    return result.success((cases, else_case))

  def if_expr_cases(self, case_keyword):
    result = ParseResult()
    cases = []
    else_case = None

    if not self.current_token.matches(Token_type_KEYWORD, case_keyword):
      return result.failure(InvalidSyntaxError(
        self.current_token.startIdx, self.current_token.endIdx,
        f"Expected '{case_keyword}'"
      ))

    result.register_incrementment()
    self.increment()

    condition = result.register(self.expr())
    if result.error: return result

    if not self.current_token.matches(Token_type_KEYWORD, 'then'):
      return result.failure(InvalidSyntaxError(
        self.current_token.startIdx, self.current_token.endIdx,
        f"Expected 'then'"
      ))

    result.register_incrementment()
    self.increment()

    if self.current_token.type == Token_type_NEWLINE:
      result.register_incrementment()
      self.increment()

      statements = result.register(self.statements())
      if result.error: return result
      cases.append((condition, statements, True))

      if self.current_token.matches(Token_type_KEYWORD, 'end'):
        result.register_incrementment()
        self.increment()
      else:
        all_cases = result.register(self.if_expr_b_or_c())
        if result.error: return result
        new_cases, else_case = all_cases
        cases.extend(new_cases)
    else:
      expr = result.register(self.expr())
      if result.error: return result
      cases.append((condition, expr, False))

      all_cases = result.register(self.if_expr_b_or_c())
      if result.error: return result
      new_cases, else_case = all_cases
      cases.extend(new_cases)

    return result.success((cases, else_case))

  def while_expr(self):
    result = ParseResult()

    if not self.current_token.matches(Token_type_KEYWORD, 'while'):
      return result.failure(InvalidSyntaxError(
        self.current_token.startIdx, self.current_token.endIdx,
        f"Expected 'while'"
      ))

    result.register_incrementment()
    self.increment()

    condition = result.register(self.expr())
    if result.error: return result

    if not self.current_token.matches(Token_type_KEYWORD, 'then'):
      return result.failure(InvalidSyntaxError(
        self.current_token.startIdx, self.current_token.endIdx,
        f"Expected 'then'"
      ))

    result.register_incrementment()
    self.increment()

    if self.current_token.type == Token_type_NEWLINE:
      result.register_incrementment()
      self.increment()

      body = result.register(self.statements())
      if result.error: return result

      if not self.current_token.matches(Token_type_KEYWORD, 'end'):
        return result.failure(InvalidSyntaxError(
          self.current_token.startIdx, self.current_token.endIdx,
          f"Expected 'end'"
        ))

      result.register_incrementment()
      self.increment()

      return result.success(WhileNode(condition, body, True))

    body = result.register(self.expr())
    if result.error: return result

    return result.success(WhileNode(condition, body, False))

  def func_def(self):
    result = ParseResult()

    if not self.current_token.matches(Token_type_KEYWORD, 'fun'):
      return result.failure(InvalidSyntaxError(
        self.current_token.startIdx, self.current_token.endIdx,
        f"Expected 'fun'"
      ))

    result.register_incrementment()
    self.increment()

    if self.current_token.type == Token_type_IDENTIFIER:
      var_name_token = self.current_token
      result.register_incrementment()
      self.increment()
      if self.current_token.type != Token_type_LPAREN:
        return result.failure(InvalidSyntaxError(
          self.current_token.startIdx, self.current_token.endIdx,
          f"Expected '('"
        ))
    else:
      var_name_token = None
      if self.current_token.type != Token_type_LPAREN:
        return result.failure(InvalidSyntaxError(
          self.current_token.startIdx, self.current_token.endIdx,
          f"Expected identifier or '('"
        ))

    result.register_incrementment()
    self.increment()
    arg_name_tokens = []

    if self.current_token.type == Token_type_IDENTIFIER:
      arg_name_tokens.append(self.current_token)
      result.register_incrementment()
      self.increment()

      while self.current_token.type == Token_type_COMMA:
        result.register_incrementment()
        self.increment()

        if self.current_token.type != Token_type_IDENTIFIER:
          return result.failure(InvalidSyntaxError(
            self.current_token.startIdx, self.current_token.endIdx,
            f"Expected identifier"
          ))

        arg_name_tokens.append(self.current_token)
        result.register_incrementment()
        self.increment()

      if self.current_token.type != Token_type_RPAREN:
        return result.failure(InvalidSyntaxError(
          self.current_token.startIdx, self.current_token.endIdx,
          f"Expected ',' or ')'"
        ))
    else:
      if self.current_token.type != Token_type_RPAREN:
        return result.failure(InvalidSyntaxError(
          self.current_token.startIdx, self.current_token.endIdx,
          f"Expected identifier or ')'"
        ))

    result.register_incrementment()
    self.increment()

    if self.current_token.type == Token_type_ARROW:
      result.register_incrementment()
      self.increment()

      body = result.register(self.expr())
      if result.error: return result

      return result.success(FuncDefNode(
        var_name_token,
        arg_name_tokens,
        body,
        False
      ))

    if self.current_token.type != Token_type_NEWLINE:
      return result.failure(InvalidSyntaxError(
        self.current_token.startIdx, self.current_token.endIdx,
        f"Expected '->' or NEWLINE"
      ))

    result.register_incrementment()
    self.increment()

    body = result.register(self.statements())
    if result.error: return result

    if not self.current_token.matches(Token_type_KEYWORD, 'end'):
      return result.failure(InvalidSyntaxError(
        self.current_token.startIdx, self.current_token.endIdx,
        f"Expected 'end'"
      ))

    result.register_incrementment()
    self.increment()

    return result.success(FuncDefNode(
      var_name_token,
      arg_name_tokens,
      body,
      True
    ))

  ###################################

 # purpose :  this is binary operation node
  def binary_operation(self, func_one, ops, func_two=None):
    if func_two == None:
      func_two = func_one

    result = ParseResult()
    left = result.register(func_one())    # left is assigned only the node
    if result.error: return result

    while self.current_token.type in ops or (self.current_token.type, self.current_token.value) in ops:
      operator_tokenen = self.current_token
      result.register_incrementment()
      self.increment()
      right = result.register(func_two())
      if result.error: return result
      left = BinaryOperationNode(left, operator_tokenen, right)

    return result.success(left) # return the success of the node, if no error

# ---------------------------------------------------------------------------------------------------------#

#######################################
# RUNTIME RESULT
#######################################

# purpose : to keep track of the current result
#           and the current error, if there is
class RTResult:
  def __init__(self):
    self.value = None
    self.error = None

# check for the error
  def register(self, result):
    self.error = result.error
    return result.value

  def success(self, value):
    self.value = value
    return self

  def failure(self, error):
    self.error = error
    return self

# ---------------------------------------------------------------------------------------------------------#

#######################################
# VALUES
#######################################

class Value:
  def __init__(self):
    self.set_pos()
    self.set_context()

  def set_pos(self, startIdx=None, endIdx=None):
    self.startIdx = startIdx
    self.endIdx = endIdx
    return self

  def set_context(self, context=None):
    self.context = context
    return self

  def added_to(self, other):
    return None, self.illegal_operation(other)

  def subbed_by(self, other):
    return None, self.illegal_operation(other)

  def multed_by(self, other):
    return None, self.illegal_operation(other)

  def dived_by(self, other):
    return None, self.illegal_operation(other)

  def powed_by(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_eq(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_ne(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_lt(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_gt(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_lte(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_gte(self, other):
    return None, self.illegal_operation(other)

  def anded_by(self, other):
    return None, self.illegal_operation(other)

  def ored_by(self, other):
    return None, self.illegal_operation(other)

  def notted(self):
    return None, self.illegal_operation(other)

  def run(self, args):
    return RTResult().failure(self.illegal_operation())

  def clone(self):
    raise Exception('No clone method defined')

  def is_true(self):
    return False

  def illegal_operation(self, other=None):
    if not other: other = self
    return RTError(
      self.startIdx, other.endIdx,
      'Illegal operation',
      self.context
    )

# purpose : for storing the Number
class Number(Value):
  def __init__(self, value):
    super().__init__()
    self.value = value

  def added_to(self, other):
    if isinstance(other, Number):
      return Number(self.value + other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def subbed_by(self, other):
    if isinstance(other, Number):
      return Number(self.value - other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def multed_by(self, other):
    if isinstance(other, Number):
      return Number(self.value * other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def dived_by(self, other):
    if isinstance(other, Number):
      if other.value == 0:
        return None, RTError(
          other.startIdx, other.endIdx,
          'Division by zero',
          self.context
        )

      return Number(self.value / other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def powed_by(self, other):
    if isinstance(other, Number):
      return Number(self.value ** other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_eq(self, other):
    if isinstance(other, Number):
      return Number(int(self.value == other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_ne(self, other):
    if isinstance(other, Number):
      return Number(int(self.value != other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_lt(self, other):
    if isinstance(other, Number):
      return Number(int(self.value < other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_gt(self, other):
    if isinstance(other, Number):
      return Number(int(self.value > other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_lte(self, other):
    if isinstance(other, Number):
      return Number(int(self.value <= other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_gte(self, other):
    if isinstance(other, Number):
      return Number(int(self.value >= other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def anded_by(self, other):
    if isinstance(other, Number):
      return Number(int(self.value and other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def ored_by(self, other):
    if isinstance(other, Number):
      return Number(int(self.value or other.value)).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def notted(self):
    return Number(1 if self.value == 0 else 0).set_context(self.context), None

  def clone(self):
    clone = Number(self.value)
    clone.set_pos(self.startIdx, self.endIdx)
    clone.set_context(self.context)
    return clone

# purspose : to return the value of the condition in the while loop
  def is_true(self):
    return self.value != 0

  def __str__(self):
    return str(self.value)

  def __repr__(self):
    return str(self.value)

Number.null = Number(0)
Number.false = Number(0)
Number.true = Number(1)
Number.math_PI = Number(math.pi)

# purspose  : for the purpose of strnig
class String(Value):
  def __init__(self, value):
    super().__init__()
    self.value = value

  def added_to(self, other):
    if isinstance(other, String):
      return String(self.value + other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def multed_by(self, other):
    if isinstance(other, Number):
      return String(self.value * other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def is_true(self):
    return len(self.value) > 0

  def clone(self):
    clone = String(self.value)
    clone.set_pos(self.startIdx, self.endIdx)
    clone.set_context(self.context)
    return clone

  def __str__(self):
    return self.value

  def __repr__(self):
    return f'"{self.value}"'

class List(Value):
  def __init__(self, elements):
    super().__init__()
    self.elements = elements

  def added_to(self, other):
    new_list = self.clone()
    new_list.elements.append(other)
    return new_list, None

  def subbed_by(self, other):
    if isinstance(other, Number):
      new_list = self.clone()
      try:
        new_list.elements.pop(other.value)
        return new_list, None
      except:
        return None, RTError(
          other.startIdx, other.endIdx,
          'Element at this index could not be removed from list because index is out of bounds',
          self.context
        )
    else:
      return None, Value.illegal_operation(self, other)

  def multed_by(self, other):
    if isinstance(other, List):
      new_list = self.clone()
      new_list.elements.extend(other.elements)
      return new_list, None
    else:
      return None, Value.illegal_operation(self, other)

  def dived_by(self, other):
    if isinstance(other, Number):
      try:
        return self.elements[other.value], None
      except:
        return None, RTError(
          other.startIdx, other.endIdx,
          'Element at this index could not be retrieved from list because index is out of bounds',
          self.context
        )
    else:
      return None, Value.illegal_operation(self, other)

  def clone(self):
    clone = List(self.elements)
    clone.set_pos(self.startIdx, self.endIdx)
    clone.set_context(self.context)
    return clone

  def __str__(self):
    return ", ".join([str(x) for x in self.elements])

  def __repr__(self):
    return f'[{", ".join([repr(x) for x in self.elements])}]'

class BaseFunction(Value):
  def __init__(self, name):
    super().__init__()
    self.name = name or "<anonymous>"

  def generate_new_context(self):
    new_context = Context(self.name, self.context, self.startIdx)
    new_context.symbol_table = SymbolTable(new_context.parent.symbol_table)
    return new_context

  def check_args(self, arg_names, args):
    result = RTResult()

    if len(args) > len(arg_names):
      return result.failure(RTError(
        self.startIdx, self.endIdx,
        f"{len(args) - len(arg_names)} too many args passed into {self}",
        self.context
      ))

    if len(args) < len(arg_names):
      return result.failure(RTError(
        self.startIdx, self.endIdx,
        f"{len(arg_names) - len(args)} too few args passed into {self}",
        self.context
      ))

    return result.success(None)

  def populate_args(self, arg_names, args, exec_ctx):
    for i in range(len(args)):
      arg_name = arg_names[i]
      arg_value = args[i]
      arg_value.set_context(exec_ctx)
      exec_ctx.symbol_table.set(arg_name, arg_value)

  def check_and_populate_args(self, arg_names, args, exec_ctx):
    result = RTResult()
    result.register(self.check_args(arg_names, args))
    if result.error: return result
    self.populate_args(arg_names, args, exec_ctx)
    return result.success(None)

# purpose : for the purpose of the fuction interpreter
class Function(BaseFunction):
  def __init__(self, name, body_node, arg_names, should_return_null):
    super().__init__(name)
    self.body_node = body_node
    self.arg_names = arg_names
    self.should_return_null = should_return_null

  def run(self, args):
    result = RTResult()
    interpreter = Interpreter()
    exec_ctx = self.generate_new_context()

    result.register(self.check_and_populate_args(self.arg_names, args, exec_ctx))
    if result.error: return result

    value = result.register(interpreter.traverse(self.body_node, exec_ctx))
    if result.error: return result
    return result.success(Number.null if self.should_return_null else value)

  def clone(self):
    clone = Function(self.name, self.body_node, self.arg_names, self.should_return_null)
    clone.set_context(self.context)
    clone.set_pos(self.startIdx, self.endIdx)
    return clone

  def __repr__(self):
    return f"<function {self.name}>"

class BuiltInFunction(BaseFunction):
  def __init__(self, name):
    super().__init__(name)

  def run(self, args):
    result = RTResult()
    exec_ctx = self.generate_new_context()

    method_name = f'run_{self.name}'
    method = getattr(self, method_name, self.no_traverse_method)

    result.register(self.check_and_populate_args(method.arg_names, args, exec_ctx))
    if result.error: return result

    return_value = result.register(method(exec_ctx))
    if result.error: return result
    return result.success(return_value)

  def no_traverse_method(self, node, context):
    raise Exception(f'No run_{self.name} method defined')

  def clone(self):
    clone = BuiltInFunction(self.name)
    clone.set_context(self.context)
    clone.set_pos(self.startIdx, self.endIdx)
    return clone

  def __repr__(self):
    return f"<built-in function {self.name}>"

  #####################################

# purpose : to perform the print function to the sceen
  def run_print(self, exec_ctx):
    print(str(exec_ctx.symbol_table.get('value')))
    return RTResult().success(Number.null)
  run_print.arg_names = ['value']

# purpose : to perform the print the return function to the sceen
  def run_print_ret(self, exec_ctx):
    return RTResult().success(String(str(exec_ctx.symbol_table.get('value'))))
  run_print_ret.arg_names = ['value']

# purpose : to perform the the assign function
  def run_input(self, exec_ctx):
    text = input()
    return RTResult().success(String(text))
  run_input.arg_names = []

  def run_input_int(self, exec_ctx):
    while True:
      text = input()
      try:
        number = int(text)
        break
      except ValueError:
        print(f"'{text}' must be an integer. Try again!")
    return RTResult().success(Number(number))
  run_input_int.arg_names = []

  def run_is_list(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), List)
    return RTResult().success(Number.true if is_number else Number.false)
  run_is_list.arg_names = ["value"]

  def run_is_function(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), BaseFunction)
    return RTResult().success(Number.true if is_number else Number.false)
  run_is_function.arg_names = ["value"]

BuiltInFunction.print       = BuiltInFunction("print")
BuiltInFunction.print_ret   = BuiltInFunction("print_ret")
BuiltInFunction.input       = BuiltInFunction("input")
BuiltInFunction.input_int   = BuiltInFunction("input_int")
BuiltInFunction.is_list     = BuiltInFunction("is_list")
BuiltInFunction.is_function = BuiltInFunction("is_function")


#######################################
# CONTEXT
#######################################

# purspoe : hold the current conext of the program
#           function if it is the function
#           otherwise the entire program
class Context:
  def __init__(self, display_name, parent=None, parent_entry_pos=None):
    self.display_name = display_name
    self.parent = parent
    self.parent_entry_pos = parent_entry_pos
    self.symbol_table = None

#######################################
# SYMBOL TABLE
#######################################

# purpose : to keep track of all the var name and its value
#           for the purpose of IDENTIFIER
#
class SymbolTable:
  def __init__(self, parent=None):
    self.symbols = {}
    self.parent = parent

# purpose : to get the value from the name
  def get(self, name):
    value = self.symbols.get(name, None)
    if value == None and self.parent:
      return self.parent.get(name)  # if the value is none return the parent's value
    return value

# purpose : take the var name to set the value
  def set(self, name, value):
    self.symbols[name] = value

# purpose: take the var name to remove the value
  def remove(self, name):
    del self.symbols[name]

# ---------------------------------------------------------------------------------------------------------#

#######################################
# INTERPRETER
#######################################

# purpose : to traverse throgh the tree that wee built up
#           with different node type
#           and determine what code should be run

class Interpreter:
  # has the traverse method to many kind of node type
  # it will take the node
  # and process the node and process of the child node
  # proces base on the node type
  def traverse(self, node, context):
    method_name = f'traverse_{type(node).__name__}' # get the type of the node and the name of the node
    method = getattr(self, method_name, self.no_traverse_method)
    return method(node, context)

# there is not visit, for the defualt
  def no_traverse_method(self, node, context):
    raise Exception(f'No traverse_{type(node).__name__} method defined')

  ###################################

  def traverse_NumberNode(self, node, context):
    return RTResult().success(
      Number(node.tok.value).set_context(context).set_pos(node.startIdx, node.endIdx)
    )

  def traverse_StringNode(self, node, context):
    return RTResult().success(
      String(node.tok.value).set_context(context).set_pos(node.startIdx, node.endIdx)
    )

  def traverse_ListNode(self, node, context):
    result = RTResult()
    elements = []

    for element_node in node.element_nodes:
      elements.append(result.register(self.traverse(element_node, context)))
      if result.error: return result

    return result.success(
      List(elements).set_context(context).set_pos(node.startIdx, node.endIdx)
    )

  def traverse_VarAccessNode(self, node, context):
    result = RTResult()
    var_name = node.var_name_token.value
    value = context.symbol_table.get(var_name)

    if not value:
      return result.failure(RTError(
        node.startIdx, node.endIdx,
        f"'{var_name}' is not defined",
        context
      ))

    value = value.clone().set_pos(node.startIdx, node.endIdx).set_context(context)
    return result.success(value)

  def traverse_VarAssignNode(self, node, context):
    result = RTResult()
    var_name = node.var_name_token.value
    value = result.register(self.traverse(node.value_node, context))
    # if there is error
    if result.error: return result
    # if there is no error
    # set the var name to its value
    context.symbol_table.set(var_name, value)
    return result.success(value)    # otherwise return sucesss result

  def traverse_BinaryOperationNode(self, node, context):
    result = RTResult()
    left = result.register(self.traverse(node.left_node, context))
    if result.error: return result
    right = result.register(self.traverse(node.right_node, context))
    if result.error: return result

    if node.operator_tokenen.type == Token_type_PLUS:
      RESULT, error = left.added_to(right)
    elif node.operator_tokenen.type == Token_type_MINUS:
      RESULT, error = left.subbed_by(right)
    elif node.operator_tokenen.type == Token_type_MUL:
      RESULT, error = left.multed_by(right)
    elif node.operator_tokenen.type == Token_type_DIV:
      RESULT, error = left.dived_by(right)
    elif node.operator_tokenen.type == Token_type_POW:
      RESULT, error = left.powed_by(right)
    elif node.operator_tokenen.type == Token_type_EE:
      RESULT, error = left.get_comparison_eq(right)
    elif node.operator_tokenen.type == Token_type_NE:
      RESULT, error = left.get_comparison_ne(right)
    elif node.operator_tokenen.type == Token_type_LT:
      RESULT, error = left.get_comparison_lt(right)
    elif node.operator_tokenen.type == Token_type_GT:
      RESULT, error = left.get_comparison_gt(right)
    elif node.operator_tokenen.type == Token_type_LTE:
      RESULT, error = left.get_comparison_lte(right)
    elif node.operator_tokenen.type == Token_type_GTE:
      RESULT, error = left.get_comparison_gte(right)
    elif node.operator_tokenen.matches(Token_type_KEYWORD, 'and'):
      RESULT, error = left.anded_by(right)
    elif node.operator_tokenen.matches(Token_type_KEYWORD, 'or'):
      RESULT, error = left.ored_by(right)

    if error:
      return result.failure(error)
    else:
      return result.success(RESULT.set_pos(node.startIdx, node.endIdx))

  def traverse_UnaryOperationNode(self, node, context):
    result = RTResult()
    number = result.register(self.traverse(node.node, context))
    if result.error: return result

    error = None

    # for the minus number only
    if node.operator_tokenen.type == Token_type_MINUS:
      number, error = number.multed_by(Number(-1))
    elif node.operator_tokenen.matches(Token_type_KEYWORD, 'not'):
      number, error = number.notted()

    if error:
      return result.failure(error)
    else:
      return result.success(number.set_pos(node.startIdx, node.endIdx))

  def traverse_IfNode(self, node, context):
    result = RTResult()

    for condition, expr, should_return_null in node.cases:
      condition_value = result.register(self.traverse(condition, context))
      if result.error: return result

      if condition_value.is_true():
        expr_value = result.register(self.traverse(expr, context))
        if result.error: return result
        return result.success(Number.null if should_return_null else expr_value)

    if node.else_case:
      expr, should_return_null = node.else_case
      expr_value = result.register(self.traverse(expr, context))
      if result.error: return result
      return result.success(Number.null if should_return_null else expr_value)

    return result.success(Number.null)

  def traverse_WhileNode(self, node, context):
    result = RTResult()
    elements = []

    while True:
      condition = result.register(self.traverse(node.condition_node, context))
      if result.error: return result

      if not condition.is_true(): break

      elements.append(result.register(self.traverse(node.body_node, context)))
      if result.error: return result

    return result.success(
      Number.null if node.should_return_null else
      List(elements).set_context(context).set_pos(node.startIdx, node.endIdx)
    )

  def traverse_FuncDefNode(self, node, context):
    result = RTResult()

    func_name = node.var_name_token.value if node.var_name_token else None
    body_node = node.body_node
    arg_names = [arg_name.value for arg_name in node.arg_name_tokens]
    func_value = Function(func_name, body_node, arg_names, node.should_return_null).set_context(context).set_pos(node.startIdx, node.endIdx)

    if node.var_name_token:
      context.symbol_table.set(func_name, func_value)

    return result.success(func_value)

  def traverse_CallNode(self, node, context):
    result = RTResult()
    args = []

    value_to_call = result.register(self.traverse(node.node_to_call, context))
    if result.error: return result
    value_to_call = value_to_call.clone().set_pos(node.startIdx, node.endIdx)

    for arg_node in node.arg_nodes:
      args.append(result.register(self.traverse(arg_node, context)))
      if result.error: return result

    return_value = result.register(value_to_call.run(args))
    if result.error: return result
    return_value = return_value.clone().set_pos(node.startIdx, node.endIdx).set_context(context)
    return result.success(return_value)

# ---------------------------------------------------------------------------------------------------------#

#######################################
# RUN
#######################################

global_symbol_table = SymbolTable()
global_symbol_table.set("NULL", Number.null)
global_symbol_table.set("FALSE", Number.false)
global_symbol_table.set("TRUE", Number.true)
global_symbol_table.set("MATH_PI", Number.math_PI)
global_symbol_table.set("print", BuiltInFunction.print)
global_symbol_table.set("print_RET", BuiltInFunction.print_ret)
global_symbol_table.set("INPUT", BuiltInFunction.input)
global_symbol_table.set("INPUT_INT", BuiltInFunction.input_int)
global_symbol_table.set("IS_LIST", BuiltInFunction.is_list)
global_symbol_table.set("IS_FUN", BuiltInFunction.is_function)


def run(fn, text):
  # Generate tokens
  lexer = Lexer(fn, text)
  tokens, error = lexer.create_tokenens()
  if error: return None, error

  # Generate AST
  parser = Parser(tokens)   # create new paerser and parse the token
  ast = parser.parse()
  if ast.error: return None, ast.error  # in case there is error

  # Run program
  # crate the interpreter instance
  interpreter = Interpreter()   # call the interpreter
  context = Context('<program>')
  context.symbol_table = global_symbol_table
  RESULT = interpreter.traverse(ast.node, context)  # pass the context through the visit method, so that the context is pass down thrugh the tree

  return RESULT.value, RESULT.error
