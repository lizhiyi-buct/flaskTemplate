"""empty message

Revision ID: a6836dd8650f
Revises: 5d5b31ab21f5
Create Date: 2023-05-01 16:23:47.767859

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a6836dd8650f'
down_revision = '5d5b31ab21f5'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('predict', schema=None) as batch_op:
        batch_op.add_column(sa.Column('is_completed', sa.Integer(), nullable=False))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('predict', schema=None) as batch_op:
        batch_op.drop_column('is_completed')

    # ### end Alembic commands ###
